// SPDX-License-Identifier: CC0-1.0

//! The PYTHON interpreter one invocation runs in, and everything that bounds and contains
//! it.
//!
//! A `python3.x` function's answer is whatever its handler returned, so the handler is
//! RUN: the deployment package's modules are compiled and executed by the interpreter this
//! module builds, in this process, and the value the handler returned is serialized by that
//! same interpreter. Nothing about a run is synthesised — a payload, a log line, an error
//! type or a duration here is the one the guest produced.
//!
//! # Containment
//!
//! The guest reaches exactly what this module registers, and nothing else:
//!
//! * the interpreter is built with the engine crate's host-reaching surface compiled OUT
//!   (`default-features = false`, see this crate's `Cargo.toml`), so no `ctypes` host
//!   loader, no host standard streams and no interpreter thread exist in the build;
//! * the native modules are an ALLOW-LIST ([`ALLOWED_NATIVE_MODULES`]) filtered out of the
//!   engine's own pool, so a module that reaches a socket, a subprocess, a memory map or a
//!   host library is never registered — it is therefore unreachable through `import`,
//!   `sys.modules`, `importlib` or `__import__`, however the guest rebinds or deletes
//!   names;
//! * the modules a Python program expects of its operating system are this module's OWN
//!   ([`host_posix_module`], [`host_thread_module`]): the environment is the invocation's
//!   own variables held in memory, the working directory is the task root the runtime
//!   reports, randomness comes from the host's own generator, and every call that would
//!   reach a host file, process or descriptor raises;
//! * `socket` is one of those: no socket EXTENSION is registered — `_socket` and `_ssl` stay
//!   off the allow-list above, so the standard library's own `socket` has nothing to be built
//!   on and no guest code can create a host socket, resolve a name or open a connection. What
//!   this runtime supplies on the import path instead ([`super::guest_http`], with the
//!   datagram half of [`super::xray_daemon`]) opens nothing: its one reachable destination is
//!   the in-process X-Ray daemon, whose datagrams travel over the SAME bounded AWS SDK seam
//!   below, and every other destination raises the error a host with no route to it answers;
//! * the only host functions the guest can call are the six on [`HOST_MODULE`]: the
//!   invocation log, a sleep clamped to what is left of the run, ONE AWS SDK call into this
//!   same emulator ([`super::aws_api`], which decides which services are reachable, signs the
//!   request itself and bounds it by what the run has left), the authentication token an
//!   SDK mints in its client, the one request a handler sends itself and the one URL it has
//!   signed. No host path, host environment variable, host credential or
//!   other function's state is reachable from guest code, and NO bound of the run — no
//!   deadline, no step count and no memory baseline — is reachable from it either: where one
//!   PHASE of a run ends and the next begins is decided by the HOST, between two calls it
//!   makes into the runtime ([`drive`]), rather than by anything the guest can reach.
//!
//! # Bounds
//!
//! Every bound lives in host code the guest cannot switch off:
//!
//! * STEPS and the DEADLINE are checked by [`Bounds::step`], which the interpreter's trace
//!   hook calls DIRECTLY: the hook is a host function ([`install_trace_hook`]) rather than
//!   Python, so no module attribute, global or default argument stands between a traced event
//!   and the bound, and the host then REMOVES `sys.settrace` / `sys.setprofile` /
//!   `sys.gettrace`, so no guest call can clear or replace it either. The stop is LATCHED:
//!   once the budget is spent the run's answer is that stop, whatever the guest goes on to
//!   return or catch.
//!   Every deadline of a run — the one the hook spends and the one the host's own watch
//!   samples — is read against the CLOCK the host hands the run ([`run_against`]), which is
//!   this host's own for every invocation this emulator serves. A run can therefore be
//!   judged against a clock whose span the host's own speed cannot spend, which is what lets
//!   a run asked about one of the bounds below be answered by THAT bound rather than by its
//!   deadline on a host slow enough to spend it.
//! * MEMORY is sampled from outside the guest — this thread's faulted pages and the
//!   process's resident size — because the interpreter exposes no allocation seam; a sample
//!   past the run's own ceiling ends the run, and so does one showing the run has grown the
//!   whole process past the process ceiling. Both are measured from what was held once the
//!   interpreter was standing and before a line of the run's own code had been read
//!   ([`Bounds::measure_memory_from_here`]), and the HOST takes that baseline: nothing the
//!   guest can call re-takes it, so the ceiling cannot be lifted by asking.
//! * A long NATIVE call emits no trace event, so neither the deadline nor a memory sample can
//!   land inside one. That is what the host's own watch ([`watch_run`]) is for: ONE thread of
//!   this process reads the same clock and the same process reading for every live run and
//!   LATCHES the stop, which the next event the interpreter delivers then raises — so one
//!   native allocation that takes the process past its ceiling ends that run without waiting
//!   for the guest to produce an event. Beside it, the caller's own host wait and
//!   abandoned-run accounting ([`super::run_invocation`]) make sure the CALLER is always
//!   answered inside its ceiling, and a run still going after its budget was spent is given
//!   up on.
//! * RECURSION is clamped on the interpreter before any guest code runs, and
//!   `sys.setrecursionlimit` is removed so the guest cannot raise it.
//! * The LOG and the RESPONSE are each capped, so a small request cannot make an
//!   unbounded answer.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime};

use rustpython_vm::function::FuncArgs;
use rustpython_vm::builtins::PyModule;
use rustpython_vm::{AsObject, Interpreter, PyObjectRef, PyRef, PyResult, Settings, VirtualMachine};

// What one run may spend and what it came to are the same for every engine, so they are
// the shared ones ([`super::outcome`]) rather than this interpreter's own.
pub use super::outcome::{Budget, Completion, Finished, Stop};

// So is the clock a run's time-based bounds are read against ([`super::run_clock`]): WHICH
// clock a run is judged by is a property of the run ([`run_against`]) rather than a call to
// the host buried inside each check.
pub use super::run_clock::Clock;

use super::aws_api::{AwsAccess, Refusal, MAX_CALLS_PER_RUN};
use super::outcome::{captured, Capture};
use super::package::PackageModule;

/// The name of the module the host functions live on, as the runtime source imports it.
const HOST_MODULE: &str = "_lambda_host";

/// Every native module this interpreter registers, out of the engine's own two pools.
///
/// This list IS the containment of the native surface: a name absent from it is dropped
/// from the interpreter's module table before any guest code runs, so no guest-side import
/// of it can succeed — not through `import`, not through `importlib`, not through
/// `__import__`, and not by dropping the module from `sys.modules` and importing it again
/// ([`prepare`] takes the builtin-module creation calls away too).
///
/// What is here is the interpreter's own machinery (the import system, the exception and
/// object model support, the bytecode and codec support) and the arithmetic, text,
/// hashing and data-structure modules the frozen standard library needs for a handler to
/// `import json`, `datetime`, `decimal`, `hashlib`, `base64`, `random`, `re` or `uuid` —
/// none of which reaches outside this process.
///
/// Deliberately absent, so that each stays unregistered: `_socket`, `_ssl`, `select`,
/// `_posixsubprocess`, `mmap`, `_ctypes`, `fcntl`, `termios`, `resource`, `pwd`, `grp`,
/// `syslog`, `_signal`, `_interpreters`, `_interpchannels`, `_interpqueues`,
/// `_testinternalcapi`, and every other module whose calls land on the host. `posix` and
/// `_thread` are absent for the same reason, and this module supplies its OWN of each
/// instead (see [`host_posix_module`], [`host_thread_module`]).
///
/// Two of the names here are registered but PATCHED before guest code runs, because the
/// frozen standard library needs the rest of each: `_io`, whose host file-opening calls
/// are replaced by refusals, and `time`, whose `sleep` is replaced by the clamped one. See
/// [`prepare`].
const ALLOWED_NATIVE_MODULES: &[&str] = &[
    // The interpreter's own machinery, which nothing runs without.
    "_abc",
    "_ast",
    "_codecs",
    "_collections",
    "_functools",
    "_imp",
    "_io",
    "_operator",
    // The opcode table `dis` and `inspect` read (`traceback` imports both). Pure data.
    "_opcode",
    "_sre",
    "_stat",
    "_string",
    // The tokenizer the standard library's own source tools use (`tokenize`, which
    // `traceback` imports). Pure text processing.
    "_tokenize",
    "_types",
    "_typing",
    "_warnings",
    "_weakref",
    "atexit",
    "errno",
    "gc",
    "itertools",
    "marshal",
    "time",
    // Arithmetic, text, hashing and data structures: computation, and nothing else.
    "_bisect",
    "_blake2",
    "_contextvars",
    "_csv",
    "_datetime",
    "_hashlib",
    "_heapq",
    "_json",
    "_md5",
    "_queue",
    "_random",
    "_sha1",
    "_sha2",
    "_sha256",
    "_sha3",
    "_sha512",
    "_statistics",
    "_struct",
    "array",
    "binascii",
    "cmath",
    "math",
    "unicodedata",
    "zlib",
];

/// The modules this emulator SUPPLIES in place of ones it does not register: the host
/// functions the runtime source calls.
const SUPPLIED_MODULES: &[&str] = &[HOST_MODULE, "posix"];

/// How many trace events pass between two readings of the whole PROCESS's resident size.
/// That reading is a request to the operating system, so it is taken often enough to catch
/// a process-wide growth inside one bounded stretch of work and seldom enough not to
/// dominate the run; the RUN's own memory is sampled at every event ([`Bounds::step`]).
const STEPS_BETWEEN_PROCESS_SAMPLES: u64 = 2_048;

/// The size of a page of this process's memory, when the host will not say.
const DEFAULT_PAGE_BYTES: u64 = 4_096;

/// The size of a page of this process's memory, which the faulted-page count is scaled by.
///
/// The host decides it: Apple silicon uses 16 KiB pages and some Linux arm64 kernels use 64 KiB.
fn page_bytes() -> u64 {
    match ::nix::unistd::sysconf(::nix::unistd::SysconfVar::PAGE_SIZE) {
        Ok(Some(size)) => u64::try_from(size).unwrap_or(DEFAULT_PAGE_BYTES),
        _ => DEFAULT_PAGE_BYTES,
    }
}

/// The resident size this WHOLE process may reach while a handler is running, whatever any
/// one run has added.
///
/// A ceiling on the PROCESS, not on one run: the run's own ceiling is its `MemorySize`, which
/// is what a run is answerable for. What this adds is that no number of concurrent runs can
/// take the emulator past it — so a run is stopped for it when the run has GROWN the process
/// past it ([`Bounds::past_process_ceiling`]), and not for a process that was already there
/// when the run began: memory a run did not allocate is not a bound that run breached, and
/// stopping it would answer the wrong caller while freeing nothing.
const PROCESS_RESIDENT_CEILING_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// One invocation to run: the package, the handler, and everything the runtime reports to
/// it.
#[derive(Debug)]
pub struct Run<'a> {
    /// The deployment package's modules.
    pub modules: &'a [PackageModule],
    /// The `Handler` configured for the function (`module.attribute`).
    pub handler: &'a str,
    /// The environment the invocation runs with: the function's own variables plus the
    /// ones the runtime sets.
    pub environment: &'a BTreeMap<String, String>,
    /// The event payload, as the caller sent it.
    pub event: Option<&'a str>,
    /// The invocation's request id.
    pub request_id: &'a str,
    /// The ARN the invocation addressed, as `context.invoked_function_arn`.
    pub invoked_function_arn: &'a str,
    /// The `ClientContext` the caller passed, already decoded to its JSON text.
    pub client_context: Option<&'a str>,
    /// The `TenantId` the caller passed.
    pub tenant_id: Option<&'a str>,
    /// What this run may spend.
    pub budget: Budget,
    /// How the handler reaches the rest of this emulator through the AWS SDK this runtime
    /// supplies it, or `None` for a run with no seam to dispatch a call on — whose
    /// `boto3` reports every call as one that was not made.
    pub aws: Option<&'a AwsAccess>,
}

/// Run `request`'s handler to completion.
///
/// Called on a thread of its own with no state guard of this service held, because the
/// interpreter is synchronous and CPU-bound.
#[must_use]
pub fn run(request: &Run<'_>) -> Finished {
    run_against(request, &Clock::host())
}

/// [`run`], against the CLOCK this run's time-based bounds are read from.
///
/// Which bound stops a run is the host's own reading, so which clock the host reads is a
/// property of the run: every invocation this emulator serves is bounded against this host's
/// own clock ([`Clock::host`], which is what [`run`] passes), and a run driven to establish
/// that ONE named bound is what stops it is bounded against a clock whose span this host's
/// speed cannot spend ([`super::run_clock`]) — so a run asked about its memory ceiling, its
/// step budget or its response cap is not answered by its deadline instead because the host
/// was slow.
#[must_use]
pub fn run_against(request: &Run<'_>, clock: &Clock) -> Finished {
    // The run's own DURATION is this host's own elapsed time whatever clock bounds the run:
    // what it reports is how long the caller waited, which is not a bound.
    let started = Instant::now();
    let capture = Arc::new(Mutex::new(Capture::new(request.budget.log_bytes)));
    let bounds = Arc::new(Bounds::new(&request.budget, clock.clone()));
    let completion = evaluate(request, &capture, &bounds);
    let elapsed = started.elapsed();
    // The stop LATCH wins over whatever the guest answered: a run whose budget was spent
    // is a stopped run even if the guest caught the exception the hook raised.
    let completion = match bounds.stopped() {
        Some(stop) => Completion::Stopped(stop),
        None => completion,
    };
    let log = captured(&capture).text.clone();
    Finished {
        completion,
        log,
        elapsed,
    }
}

/// Build the interpreter, install the host surface, and drive the runtime source.
fn evaluate(request: &Run<'_>, capture: &Arc<Mutex<Capture>>, bounds: &Arc<Bounds>) -> Completion {
    let input = match runtime_input(request) {
        Ok(input) => input,
        Err(error) => return Completion::Internal(error),
    };
    let contained = Arc::new(AtomicBool::new(false));
    let interpreter = interpreter(&contained);
    // A run only happens when the native surface really was narrowed to the allow-list. If
    // it could not be, nothing is run at all: an uncontained interpreter is never handed
    // guest code.
    if !contained.load(Ordering::SeqCst) {
        return Completion::Internal(
            "the interpreter's native modules could not be narrowed to the allow-list".to_string(),
        );
    }
    // The hook the bounds are spent through, held HERE: it stays installed and answering
    // until the interpreter has been finalized, because finalization runs the guest's own
    // `atexit` callbacks and those frames are traced like every other guest frame.
    let held: TraceCell = ::std::rc::Rc::new(::std::cell::RefCell::new(None));
    // The host's own watch over this run, sampling the clock from OUTSIDE the guest for as
    // long as the run lasts ([`watch_run`]).
    watch_run(bounds);
    let completion = interpreter.enter(|vm| {
        match prepare(vm, request, capture, bounds)
            .and_then(|()| drive(vm, &input, bounds, &held))
        {
            Ok(result) => result,
            Err(exception) => Completion::Internal(formatted_exception(vm, &exception)),
        }
    });
    // The interpreter is FINALIZED once the run is over, which is what releases the objects
    // it built: an interpreter that is only dropped leaves its own reference cycles — the
    // module table, the code objects, the exception types — uncollected, so each invocation
    // would hold its interpreter's memory for the life of this process however short the run
    // was. Finalizing it is also what the language itself does at the end of a program, so a
    // handler's `atexit` callback runs the way it runs on the real runtime, and it runs
    // against the SAME bounds the handler did — the hook is still installed and still
    // answering itself as each frame's local tracer, the host watch below is still sampling
    // the clock, and a stop already latched stays latched — so nothing of the run outlasts
    // the run's budget.
    interpreter.finalize(None);
    // Only now: the hook and this cell hold each other, so emptying it is what lets the one
    // the run installed go.
    held.borrow_mut().take();
    completion
}

/// How often the host's own watch samples a run's deadline.
///
/// A long NATIVE call emits no traced event, so a deadline checked only from the trace hook is
/// one the guest decides the timing of. The watch reads the same clock from a thread of its
/// own, so a run that stops producing events is still noticed and still stopped.
const WATCH_INTERVAL: Duration = Duration::from_millis(50);

/// Have the host's own watch sample this run's deadline while the run lasts.
///
/// The watch is ONE thread for the whole process, not one per run: it holds a WEAK reference
/// to each run's bounds, so a run it is watching is one that is still going and a run that
/// ended simply falls out of the list. The thread is started with the list, on the first run
/// of this process.
fn watch_run(bounds: &Arc<Bounds>) {
    let mut watched = watched_runs()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    watched.retain(|run| run.strong_count() > 0);
    watched.push(Arc::downgrade(bounds));
    drop(watched);
    start_watch_thread();
}

/// The bounds of every run the watch is sampling.
fn watched_runs() -> &'static Mutex<Vec<::std::sync::Weak<Bounds>>> {
    static WATCHED: ::std::sync::OnceLock<Mutex<Vec<::std::sync::Weak<Bounds>>>> =
        ::std::sync::OnceLock::new();
    WATCHED.get_or_init(|| Mutex::new(Vec::new()))
}

/// Start the one watch thread of this process, if it is not already running.
fn start_watch_thread() {
    static STARTED: ::std::sync::OnceLock<()> = ::std::sync::OnceLock::new();
    STARTED.get_or_init(|| {
        let _ = ::std::thread::Builder::new()
            .name("lambda-run-watch".to_string())
            .spawn(|| loop {
                ::std::thread::sleep(WATCH_INTERVAL);
                // The live runs are copied out under the lock and sampled outside it, so a
                // sample never holds the list against the next run that starts.
                let live: Vec<Arc<Bounds>> = {
                    let mut watched = watched_runs()
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    watched.retain(|run| run.strong_count() > 0);
                    watched.iter().filter_map(::std::sync::Weak::upgrade).collect()
                };
                for bounds in live {
                    bounds.watch();
                }
            });
    });
}

/// The interpreter itself: the frozen standard library and the ALLOW-LISTED native
/// modules, and nothing else.
///
/// `contained` reports whether the narrowing really happened: the module table is reached
/// through the one mutable borrow the initialisation hook has, and a run is refused rather
/// than served if that borrow is not available.
fn interpreter(contained: &Arc<AtomicBool>) -> Interpreter {
    let mut settings = Settings::default();
    // No host path is searched for a module: the standard library is the frozen one and
    // the package's modules are resolved by the runtime's own finder.
    settings.path_list = Vec::new();
    settings.ignore_environment = true;
    settings.isolated = true;
    settings.write_bytecode = false;
    settings.quiet = true;
    // Nothing of the HOST is processed on the way up: no signal handler is installed for
    // this process, and the `site` module — which exists to read host directories — is not
    // imported.
    settings.install_signal_handlers = false;
    settings.import_site = false;
    let contained = Arc::clone(contained);
    Interpreter::with_init(settings, move |vm| {
        let allowed: Vec<&'static ::rustpython_vm::builtins::PyModuleDef> =
            ::rustpython_stdlib::stdlib_module_defs(&vm.ctx)
                .into_iter()
                .filter(|def| ALLOWED_NATIVE_MODULES.contains(&def.name.as_str()))
                .collect();
        // The `posix` module a run is built on is THIS crate's (see [`super::host_posix`]):
        // the import system requires a module of that name to exist, and the engine's own
        // — whose calls are the host's — needs an engine feature this crate does not build
        // at all, so there is none in the binary to register.
        let posix = super::host_posix::module::module_def(&vm.ctx);
        let Some(state) = ::std::rc::Rc::get_mut(&mut vm.state) else {
            return;
        };
        // The native surface of the interpreter, narrowed to the allow-list: the engine's
        // own core modules that are not on it — every one whose calls land on the host —
        // are dropped from the table, so no guest import of them can succeed.
        state
            .module_defs
            .retain(|name, _| ALLOWED_NATIVE_MODULES.contains(name));
        for def in allowed {
            state.module_defs.insert(def.name.as_str(), def);
        }
        state.module_defs.insert("posix", posix);
        state.frozen.extend(::rustpython_pylib::FROZEN_STDLIB);
        contained.store(true, Ordering::SeqCst);
    })
}

/// Install everything the guest sees of the host, before any guest code runs: the modules
/// this emulator supplies in place of the ones it does not register, the refusals that
/// replace the host-reaching calls of the two it does, the bounds, and the log the
/// handler's output goes into.
fn prepare(
    vm: &VirtualMachine,
    request: &Run<'_>,
    capture: &Arc<Mutex<Capture>>,
    bounds: &Arc<Bounds>,
) -> PyResult<()> {
    let modules = vm.sys_module.as_object().get_attr("modules", vm)?;
    // Every allow-listed native module is created ONCE, here, so each is the single
    // patched object `sys.modules` holds from now on. Nothing is imported from a host
    // path: these are the interpreter's own built-in modules.
    for name in ALLOWED_NATIVE_MODULES.iter().chain(["posix"].iter()) {
        let _ = vm.import(*name, 0);
    }
    refuse_host_file_access(vm)?;
    narrow_posix(vm)?;
    // The DATA `os` re-exports from the module it is built on — the `stat_result` type and
    // the `open` flags — installed before any import of `os`, because that is where `os`
    // takes them from ([`super::host_posix::install_data_surface`]).
    super::host_posix::install_data_surface(vm)?;
    // The codec registry, AFTER the two installations above: whatever it imports then sees
    // the same patched `io` and the same complete `os` every later import sees, so it can
    // hold neither an unpatched way of opening a host file nor a half-built `os`.
    install_codec_registry(vm)?;
    install_environment(vm, request.environment)?;
    refuse_guest_threads(vm)?;
    clamp_guest_sleep(vm, bounds)?;
    // A handler runs on Lambda's Linux, whatever this emulator is built on. The standard
    // library branches on `sys.platform` (`urllib` imports the macOS `_scproxy` on
    // `darwin`), so the guest sees the platform it runs on, not the host's.
    vm.sys_module
        .as_object()
        .set_attr("platform", vm.ctx.new_str("linux"), vm)?;
    // Put the host module in `sys.modules` under the name the runtime source imports it
    // by. The import machinery answers from there first, and the name is NOT in the native
    // module table ([`ALLOWED_NATIVE_MODULES`]), so there is no other module of that name
    // to be had: a guest that drops this one cannot import a second.
    modules.set_item(
        HOST_MODULE,
        host_module(vm, capture, bounds, request.aws).into(),
        vm,
    )?;
    // A built-in module can no longer be CREATED: every one the guest can reach is the
    // patched object already in `sys.modules`, and a guest that drops one cannot build
    // itself an unpatched replacement. `is_builtin` stays, because the import system asks
    // it on every import; what is taken away is the creation itself.
    let importer = vm.import("_imp", 0)?;
    let refusing = vm.new_function("create_builtin", |_args: FuncArgs, vm: &VirtualMachine| {
        Err::<PyObjectRef, _>(vm.new_import_error(
            "a Lambda function in this emulator cannot create a built-in module".to_owned(),
            vm.ctx.new_str("<builtin>"),
        ))
    });
    for name in [
        "create_builtin",
        "exec_builtin",
        "create_dynamic",
        "exec_dynamic",
    ] {
        let _ = importer.set_attr(name, refusing.clone(), vm);
    }
    // `os` decides which operating-system module to build itself on from this tuple, so it
    // names the one this emulator supplies.
    let names: Vec<PyObjectRef> = SUPPLIED_MODULES
        .iter()
        .chain(ALLOWED_NATIVE_MODULES.iter())
        .map(|name| vm.ctx.new_str(*name).into())
        .collect();
    vm.sys_module
        .as_object()
        .set_attr("builtin_module_names", vm.ctx.new_tuple(names), vm)?;
    clamp_recursion(vm, request.budget.recursion_limit)
}

/// Install the invocation's own environment into the module `os` builds `os.environ` out
/// of.
///
/// Done HERE rather than from the runtime source, so the guest's environment is the
/// host's statement of it and no Python code of this emulator's own needs an
/// operating-system module. The mapping is the one [`super::host_posix`] created for this
/// interpreter and starts EMPTY, so what a handler reads is the function's own variables
/// and nothing of this process's environment. The entries are bytes, which is the form
/// `os` decodes a POSIX environment from.
fn install_environment(vm: &VirtualMachine, variables: &BTreeMap<String, String>) -> PyResult<()> {
    let system = vm.import("posix", 0)?;
    let environ = system.get_attr("environ", vm)?;
    for (name, value) in variables {
        let key: PyObjectRef = vm.ctx.new_bytes(name.as_bytes().to_vec()).into();
        let held: PyObjectRef = vm.ctx.new_bytes(value.as_bytes().to_vec()).into();
        environ.set_item(&*key, held, vm)?;
    }
    Ok(())
}

/// Put the standard library's CODEC REGISTRY on its feet, the way a Python interpreter's own
/// start-up does, before any guest code runs.
///
/// Every text encoding a program names by its name — `bytes.decode("cp437")`,
/// `str.encode("cp1252")`, `codecs.lookup(..)` — is resolved through the search function the
/// standard library's `encodings` package registers while it is being imported, and the
/// interpreter resolves only a handful of encodings on its own without it. The engine crate
/// performs that import at start-up only under its own `encodings` feature, which this crate
/// does not take (it takes `compiler` alone, see this crate's `Cargo.toml`, so the rest of
/// that crate's start-up surface stays out of the build), so the import is performed here
/// instead: without it a module of the standard library that names one of the other encodings
/// — the archive reader decoding an entry's name is one — raises where it reads cleanly on a
/// managed runtime, and whether it raised would depend on whether the handler happened to
/// import `encodings` itself.
///
/// The package is the FROZEN standard library's own ([`interpreter`]), so no host path is read
/// to import it and nothing of the host becomes reachable through it.
fn install_codec_registry(vm: &VirtualMachine) -> PyResult<()> {
    vm.import("encodings", 0).map(|_| ())
}

/// Take away every way of opening a HOST file: the TYPE that opens one, and each of the three
/// names a Python program reaches it through.
///
/// The refusal is put on the type's own construction, not only on the names: a type that has
/// been created stays in the interpreter's type graph and is reachable by traversal from
/// `object`, so replacing `_io.FileIO` alone would leave the type that opens the file one
/// `__subclasses__` walk away. With `__new__` and `__init__` refusing, an instance obtained
/// however it was reached opens nothing. The guest sees the names it expects — `open`,
/// `io.open`, `_io.FileIO` — and the call raises. The rest of `_io` (the in-memory streams the
/// standard library is built on) is left exactly as it is.
fn refuse_host_file_access(vm: &VirtualMachine) -> PyResult<()> {
    // `io` is loaded FIRST, so the standard library's own stream classes are bound and
    // registered from the real module before anything on it is replaced.
    let _ = vm.import("io", 0);
    let refusing = vm.new_function("open", |_args: FuncArgs, vm: &VirtualMachine| {
        Err::<PyObjectRef, _>(vm.new_os_error(
            "a Lambda function in this emulator cannot open a file of the host".to_owned(),
        ))
    });
    // The TYPE, before the name it is read off is replaced: its construction is what opens a
    // host file, so that is where the refusal goes.
    if let Ok(raw_file) = vm.import("_io", 0).and_then(|module| module.get_attr("FileIO", vm)) {
        for slot in ["__new__", "__init__"] {
            let _ = raw_file.set_attr(slot, refusing.clone(), vm);
        }
    }
    let targets: [(&str, &[&str]); 2] = [
        ("_io", &["open", "open_code", "FileIO"]),
        ("io", &["open", "open_code", "FileIO"]),
    ];
    for (module, attributes) in targets {
        if let Ok(module) = vm.import(module, 0) {
            for attribute in attributes {
                let _ = module.set_attr(*attribute, refusing.clone(), vm);
            }
        }
    }
    let builtins = vm.builtins.as_object();
    for attribute in ["open", "open_code"] {
        let _ = builtins.set_attr(attribute, refusing.clone(), vm);
    }
    Ok(())
}

/// Replace the interpreter's own `sleep` with one clamped to what the run has left, on the
/// single clock module `sys.modules` holds.
fn clamp_guest_sleep(vm: &VirtualMachine, bounds: &Arc<Bounds>) -> PyResult<()> {
    let sleep_bounds = Arc::clone(bounds);
    let clamped = vm.new_function("sleep", move |args: FuncArgs, vm: &VirtualMachine| {
        let seconds: f64 = args.bind(vm)?;
        bounded_sleep(&sleep_bounds, seconds);
        Ok::<PyObjectRef, ::rustpython_vm::builtins::PyBaseExceptionRef>(vm.ctx.none())
    });
    if let Ok(clock) = vm.import("time", 0) {
        let _ = clock.set_attr("sleep", clamped, vm);
    }
    Ok(())
}

/// Wait for `seconds`, but never past what the run has left: a blocking wait the guest
/// asked for cannot outlast the run's deadline.
///
/// A value that is not a finite number of seconds is no wait at all, so nothing is waited
/// for: the guest decides how long to ask for, never how the host reads it.
fn bounded_sleep(bounds: &Arc<Bounds>, seconds: f64) {
    if !seconds.is_finite() || seconds <= 0.0 {
        return;
    }
    let asked = Duration::from_secs_f64(seconds.min(MAX_GUEST_SLEEP_SECONDS));
    ::std::thread::sleep(asked.min(bounds.remaining()));
}

/// The longest a guest may ask to wait for in one call, before the run's own remaining
/// time clamps it further.
const MAX_GUEST_SLEEP_SECONDS: f64 = 3_600.0;

/// Evaluate the runtime source and read its answer back, one PHASE at a time.
///
/// The runtime's own setup — its imports and definitions — runs FIRST, with no bound
/// spending against it, because none of it is the caller's work. Then the two calls the
/// runtime leaves behind ([`super::bootstrap`]) are made in turn: `__lambda_prepare` runs
/// the INIT phase — this runtime's installation, the import of the handler's module and the
/// resolution of its entry point — and `__lambda_invoke` runs the handler itself. The HOST
/// decides where the boundary is: it begins the invoke phase between the two calls
/// ([`Bounds::start`]), which is what starts the deadline the function's `Timeout` states
/// and what the handler is told it has left. Nothing the guest can call moves it.
fn drive(
    vm: &VirtualMachine,
    input: &str,
    bounds: &Arc<Bounds>,
    held: &TraceCell,
) -> PyResult<Completion> {
    let scope = vm.new_scope_with_builtins();
    let code = vm
        .compile(
            super::bootstrap::RUNTIME_SOURCE,
            ::rustpython_vm::compiler::Mode::Exec,
            "<lambda-runtime>".to_owned(),
        )
        .map_err(|error| {
            vm.new_runtime_error(format!("the Lambda runtime source does not compile: {error}"))
        })?;
    vm.run_code_obj(code, scope.clone())?;
    let prepare = scope.globals.get_item("__lambda_prepare", vm)?;
    let invoke = scope.globals.get_item("__lambda_invoke", vm)?;
    install_trace_hook(vm, bounds, held)?;
    // The interpreter is standing and nothing of the RUN's own code has been read yet, so
    // this is where the memory the run itself takes is measured from. The host takes this
    // baseline, never the guest.
    bounds.measure_memory_from_here();
    // The INIT phase, under the span an initialization is allowed rather than the
    // function's `Timeout`. It answers nothing when the environment is ready, and the
    // document of a failure that belongs to no invocation when it is not.
    let prepared = prepare.call((vm.ctx.new_str(input),), vm)?;
    let prepared: String = prepared.try_into_value(vm)?;
    if !prepared.is_empty() {
        return Ok(read_completion(&prepared, bounds.response_bytes));
    }
    // The INVOKE phase begins HERE, which is the handler's own run: from this instant the
    // deadline is the function's `Timeout`, and the instant it falls at is what the handler
    // is told it has left.
    let deadline_epoch_ms = bounds.start();
    let answered = invoke.call((vm.ctx.new_int(deadline_epoch_ms),), vm)?;
    let text: String = answered.try_into_value(vm)?;
    Ok(read_completion(&text, bounds.response_bytes))
}

/// The runtime's answer, as this module reports it.
fn read_completion(answered: &str, response_bytes: usize) -> Completion {
    let Ok(document) = ::serde_json::from_str::<::serde_json::Value>(answered) else {
        return Completion::Internal("the runtime answered with a document that is not JSON".into());
    };
    let member = |name: &str| {
        document
            .get(name)
            .and_then(::serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    match document.get("outcome").and_then(::serde_json::Value::as_str) {
        Some("returned") => {
            let payload = member("payload");
            if payload.len() > response_bytes {
                return Completion::Stopped(Stop::Response);
            }
            Completion::Returned(payload)
        }
        Some("raised") => Completion::Raised {
            error_type: member("errorType"),
            error_message: member("errorMessage"),
            stack_trace: document
                .get("stackTrace")
                .and_then(::serde_json::Value::as_array)
                .map(|frames| {
                    frames
                        .iter()
                        .filter_map(::serde_json::Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
        },
        _ => Completion::Internal("the runtime answered with no outcome".into()),
    }
}

/// The JSON document the runtime source is handed.
fn runtime_input(request: &Run<'_>) -> Result<String, String> {
    let modules: ::serde_json::Map<String, ::serde_json::Value> = request
        .modules
        .iter()
        .map(|module| {
            (
                module.path.clone(),
                ::serde_json::Value::String(module.source.clone()),
            )
        })
        .collect();
    let environment: ::serde_json::Map<String, ::serde_json::Value> = request
        .environment
        .iter()
        .map(|(name, value)| (name.clone(), ::serde_json::Value::String(value.clone())))
        .collect();
    let client_context = match request.client_context {
        Some(text) => ::serde_json::from_str::<::serde_json::Value>(text)
            .unwrap_or(::serde_json::Value::Null),
        None => ::serde_json::Value::Null,
    };
    let document = ::serde_json::json!({
        "handler": request.handler,
        "modules": modules,
        "runtimeModules": runtime_modules(),
        "runtimeStdlib": runtime_stdlib(),
        "environment": environment,
        "event": request.event,
        "requestId": request.request_id,
        "invokedFunctionArn": request.invoked_function_arn,
        "clientContext": client_context,
        "tenantId": request.tenant_id,
    });
    ::serde_json::to_string(&document).map_err(|error| error.to_string())
}

/// The modules this RUNTIME supplies on the handler's import path, beside the deployment
/// package's own: the AWS SDK a managed `python3.x` runtime ships
/// ([`super::aws_sdk_guest`]).
fn runtime_modules() -> ::serde_json::Map<String, ::serde_json::Value> {
    super::aws_sdk_guest::runtime_modules()
        .into_iter()
        .map(|module| (module.path, ::serde_json::Value::String(module.source)))
        .collect()
}

/// The modules of the STANDARD LIBRARY this runtime supplies in place of the frozen ones:
/// the socket module a Python program expects of its operating system and the HTTP client
/// built on one ([`super::guest_http`]). Resolved before the frozen standard library and
/// after the deployment package's own entries, which is where a runtime's own standard
/// library sits.
fn runtime_stdlib() -> ::serde_json::Map<String, ::serde_json::Value> {
    super::guest_http::runtime_stdlib_modules()
        .into_iter()
        .map(|module| (module.path, ::serde_json::Value::String(module.source)))
        .collect()
}

/// Install the trace hook the step and deadline bounds are checked from, then take away
/// every way the guest could clear or replace it.
///
/// The hook is a HOST function, not Python: what the interpreter calls on every traced event
/// is [`Bounds::step`] itself, reached through no name the guest can rebind — there is no
/// module attribute, no global and no default argument between the event and the bound. A
/// trace function has to answer the local tracer to use, and the local tracer is this same
/// function, so the one it answers is read from `held` — the cell [`evaluate`] owns, which
/// stays filled until the interpreter has been FINALIZED, because a guest `atexit` callback
/// runs during finalization and its line events have to be traced too.
fn install_trace_hook(
    vm: &VirtualMachine,
    bounds: &Arc<Bounds>,
    held: &TraceCell,
) -> PyResult<()> {
    let answered = ::std::rc::Rc::clone(held);
    let step_bounds = Arc::clone(bounds);
    let hook = vm.new_function(
        "__lambda_trace",
        move |_args: FuncArgs, vm: &VirtualMachine| {
            if let Some(stop) = step_bounds.step() {
                return Err(vm.new_runtime_error(stop.reason().to_owned()));
            }
            // The local tracer for the frame this event belongs to is this same hook, so
            // every event of every frame spends the budget.
            let held = answered.borrow();
            Ok(held.clone().unwrap_or_else(|| vm.ctx.none()))
        },
    );
    let hook: PyObjectRef = hook.into();
    *held.borrow_mut() = Some(hook.clone());
    let setter = vm.sys_module.as_object().get_attr("settrace", vm)?;
    setter.call((hook,), vm)?;
    // With these gone, no guest call can reach the interpreter's trace slot at all: the
    // hook that spends the budget is the only one there will be for this run.
    for name in ["settrace", "setprofile", "gettrace", "getprofile"] {
        let _ = vm.sys_module.as_object().del_attr(name, vm);
    }
    Ok(())
}

/// Where the installed trace hook is held, for the life of one run.
///
/// The hook answers ITSELF as each frame's local tracer, so this cell holds the hook and the
/// hook holds this cell. [`evaluate`] owns it and empties it once the interpreter has been
/// finalized — not before, because a guest `atexit` callback runs during finalization and has
/// to be traced like every other guest frame — which is what keeps one run's hook from
/// outliving the run that installed it.
type TraceCell = ::std::rc::Rc<::std::cell::RefCell<Option<PyObjectRef>>>;

/// Clamp how deep the guest may recurse, and take away the call that would raise it.
fn clamp_recursion(vm: &VirtualMachine, limit: usize) -> PyResult<()> {
    let setter = vm.sys_module.as_object().get_attr("setrecursionlimit", vm)?;
    setter.call((limit,), vm)?;
    let _ = vm.sys_module.as_object().del_attr("setrecursionlimit", vm);
    Ok(())
}

/// The module the host functions live on, and the whole of what guest code can call of
/// this process: `log`, `sleep`, `aws`, `db_auth_token`, `request`, `presign` and
/// `pagination` — the invocation log, a sleep clamped to what the run has left, ONE AWS SDK
/// call into this same emulator, the authentication token an SDK mints in its client, the one
/// request a handler sends itself, the one URL it has signed, and what the called service's
/// own model states about paging one operation through. Nothing else of the host is
/// reachable from guest code, and nothing here moves a bound.
fn host_module(
    vm: &VirtualMachine,
    capture: &Arc<Mutex<Capture>>,
    bounds: &Arc<Bounds>,
    aws: Option<&AwsAccess>,
) -> PyRef<PyModule> {
    let module = vm.new_module(HOST_MODULE, vm.ctx.new_dict(), None);
    let log_capture = Arc::clone(capture);
    set_function(vm, &module, "log", move |args: FuncArgs, vm: &VirtualMachine| {
        let text: String = args.bind(vm)?;
        captured(&log_capture).write(&text);
        Ok(vm.ctx.none())
    });
    let sleep_bounds = Arc::clone(bounds);
    set_function(
        vm,
        &module,
        "sleep",
        move |args: FuncArgs, vm: &VirtualMachine| {
            let seconds: f64 = args.bind(vm)?;
            // A sleep the guest asked for never outlasts the run: it is clamped to what
            // the run has left, so no guest call can wait past the deadline.
            bounded_sleep(&sleep_bounds, seconds);
            Ok(vm.ctx.none())
        },
    );
    // Where the INIT phase ends and the handler's own run begins is the HOST's to decide,
    // so there is no name for it on this module at all: [`drive`] makes the two calls the
    // runtime leaves behind and begins the invoke phase between them ([`Bounds::start`]).
    // The hook that spends the step, deadline and memory budget is NOT here either: the
    // interpreter calls it directly ([`install_trace_hook`]), so there is no name on this
    // module for a guest to rebind and no way for guest code to reach the bounds at all.
    // The ONE call a handler reaches another service of this emulator by. The service
    // table, the signature and the destination are all [`super::aws_api`]'s, so what the
    // guest decides is only which operation of which reachable service it asks for and
    // what it sends; the two bounds that are this run's — how many calls it may make and
    // how long one may take — are held here, where no guest code can reach them.
    let access = aws.cloned();
    let call_bounds = Arc::clone(bounds);
    // How many calls this RUN has made, counted once for every way it can make one: an SDK
    // call and a request the handler sends itself spend the same budget.
    let made = Arc::new(AtomicU64::new(0));
    let counted = Arc::clone(&made);
    set_function(
        vm,
        &module,
        "aws",
        move |args: FuncArgs, vm: &VirtualMachine| {
            let (service, operation, region, body): (String, String, String, String) =
                args.bind(vm)?;
            // A run whose budget is already spent makes no further call: the latched stop
            // is its answer, so work it starts afterwards is work nothing will report.
            if let Some(stop) = call_bounds.stopped() {
                return Err(vm.new_runtime_error(stop.reason().to_owned()));
            }
            if counted.fetch_add(1, Ordering::Relaxed) >= MAX_CALLS_PER_RUN as u64 {
                return Err(vm.new_runtime_error(format!(
                    "one invocation may make at most {MAX_CALLS_PER_RUN} AWS SDK calls"
                )));
            }
            let Some(access) = access.as_ref() else {
                return Ok(vm
                    .ctx
                    .new_str(refusal_answer(
                        "not_made",
                        "this invocation was given no AWS SDK seam, so no call was made",
                    ))
                    .into());
            };
            let remaining = call_bounds.remaining();
            if remaining.is_zero() {
                return Err(vm.new_runtime_error(Stop::Deadline.reason().to_owned()));
            }
            let answered = access.call(&service, &operation, &region, &body, remaining);
            Ok(vm.ctx.new_str(call_answer(&answered)).into())
        },
    );
    // The AUTHENTICATION TOKEN an SDK mints in the CLIENT for a database login
    // ([`super::db_auth_token`]). It is not a call — nothing is dispatched and no service
    // answers — but it is signed with the credential this seam holds, which is why the host
    // mints it and the guest only states which endpoint and which database user it is for.
    // It counts against the same per-run budget as a call, so no run can mint tokens
    // without end.
    let token_access = aws.cloned();
    let token_bounds = Arc::clone(bounds);
    let minted = Arc::clone(&made);
    set_function(
        vm,
        &module,
        "db_auth_token",
        move |args: FuncArgs, vm: &VirtualMachine| {
            let (host, port, user, region): (String, u16, String, String) = args.bind(vm)?;
            if let Some(stop) = token_bounds.stopped() {
                return Err(vm.new_runtime_error(stop.reason().to_owned()));
            }
            if minted.fetch_add(1, Ordering::Relaxed) >= MAX_CALLS_PER_RUN as u64 {
                return Err(vm.new_runtime_error(format!(
                    "one invocation may make at most {MAX_CALLS_PER_RUN} AWS SDK calls"
                )));
            }
            let Some(access) = token_access.as_ref() else {
                return Ok(vm
                    .ctx
                    .new_str(refusal_answer(
                        "not_made",
                        "this invocation was given no AWS SDK seam, so no authentication \
                         token was minted",
                    ))
                    .into());
            };
            let answer = match access.db_auth_token(&host, port, &user, &region) {
                Ok(token) => ::serde_json::json!({ "token": token }).to_string(),
                Err(Refusal::NotMade(detail)) => refusal_answer("not_made", &detail),
                Err(_other) => refusal_answer(
                    "not_made",
                    "the authentication token could not be minted",
                ),
            };
            Ok(vm.ctx.new_str(answer).into())
        },
    );
    // The ONE request a handler SENDS ITSELF, for a service whose data plane is reached
    // over HTTP rather than through an SDK operation. The address it may name, the
    // signature and the destination are all [`super::guest_http`]'s; the two bounds that
    // are this run's — the call count it shares with the SDK seam above and how long one
    // request may take — are held here, where no guest code can reach them.
    let sending = aws.cloned();
    let request_bounds = Arc::clone(bounds);
    let requested = made;
    set_function(
        vm,
        &module,
        "request",
        move |args: FuncArgs, vm: &VirtualMachine| {
            let (method, url, headers, body): (String, String, String, String) = args.bind(vm)?;
            if let Some(stop) = request_bounds.stopped() {
                return Err(vm.new_runtime_error(stop.reason().to_owned()));
            }
            if requested.fetch_add(1, Ordering::Relaxed) >= MAX_CALLS_PER_RUN as u64 {
                return Err(vm.new_runtime_error(format!(
                    "one invocation may make at most {MAX_CALLS_PER_RUN} calls into this \
                     emulator"
                )));
            }
            let Some(access) = sending.as_ref() else {
                return Ok(vm
                    .ctx
                    .new_str(refusal_answer(
                        "not_made",
                        "this invocation was given no seam to send a request on, so none was \
                         sent",
                    ))
                    .into());
            };
            let remaining = request_bounds.remaining();
            if remaining.is_zero() {
                return Err(vm.new_runtime_error(Stop::Deadline.reason().to_owned()));
            }
            let answered = access.send(&method, &url, &headers, &body, remaining);
            Ok(vm.ctx.new_str(request_answer(&answered)).into())
        },
    );
    // The URL a handler asks the SDK to PRESIGN. Nothing is sent, so this spends neither
    // the call count nor the deadline; the operation's own binding and the signature are
    // host code's ([`super::aws_api::presigned`]), so what the guest decides is only which
    // operation of which reachable service the URL names.
    let presigning = aws.cloned();
    set_function(
        vm,
        &module,
        "presign",
        move |args: FuncArgs, vm: &VirtualMachine| {
            let (service, operation, region, body, expires): (
                String,
                String,
                String,
                String,
                u64,
            ) = args.bind(vm)?;
            let Some(access) = presigning.as_ref() else {
                return Ok(vm
                    .ctx
                    .new_str(refusal_answer(
                        "not_made",
                        "this invocation was given no AWS SDK seam, so no URL was signed",
                    ))
                    .into());
            };
            let signed = access.presigned(
                &service,
                &operation,
                &region,
                &body,
                Duration::from_secs(expires),
            );
            let answered = match signed {
                Ok(url) => ::serde_json::json!({ "url": url }).to_string(),
                Err(refused) => refusal_answer_for(&refused),
            };
            Ok(vm.ctx.new_str(answered).into())
        },
    );
    // What the SDK's own PAGINATOR for one operation drives: the input member a continuation
    // value is sent back in, the output member it arrives in, the member a page's size is
    // asked for and the members a page's results arrive in, read out of the CALLED service's
    // own model ([`super::sdk_model`]). Nothing is dispatched and no service answers, so this
    // spends neither the call count nor the deadline; what the guest decides is only which
    // operation of which service it asks about.
    set_function(
        vm,
        &module,
        "pagination",
        move |args: FuncArgs, vm: &VirtualMachine| {
            let (service, operation): (String, String) = args.bind(vm)?;
            Ok(vm
                .ctx
                .new_str(super::aws_sdk_guest::pagination_config(&service, &operation))
                .into())
        },
    );
    module
}

/// What the guest reads back from one AWS SDK call: the service's own answer, or the
/// reason no call was made.
///
/// A document rather than a Python object, for the reason the runtime's own protocol is one
/// ([`super::bootstrap`]): the host's interpreter surface stays a string in and a string
/// out.
fn call_answer(answered: &Result<super::aws_api::Answer, Refusal>) -> String {
    match answered {
        Ok(answer) => ::serde_json::json!({
            "status": answer.status,
            "body": answer.body,
            "errorType": answer.error_type,
            "requestId": answer.request_id,
            "streams": answer.streams,
            "blobs": answer.blobs,
        })
        .to_string(),
        Err(refused) => refusal_answer_for(refused),
    }
}

/// The answer for a call or a signature this seam did not make, naming which refusal it
/// was. Shared by every host call the guest makes, so one refusal is reported one way.
fn refusal_answer_for(refused: &Refusal) -> String {
    match refused {
        Refusal::UnknownService => refusal_answer(
            "unknown_service",
            &super::aws_api::reachable_services().join(", "),
        ),
        Refusal::UnknownOperation => refusal_answer("unknown_operation", ""),
        Refusal::TimedOut => refusal_answer("timed_out", ""),
        Refusal::RequestTooLarge => refusal_answer(
            "not_made",
            &format!(
                "the request document is larger than the {} bytes one AWS SDK call may carry",
                super::aws_api::MAX_REQUEST_BYTES
            ),
        ),
        Refusal::ResponseTooLarge => refusal_answer(
            "not_made",
            &format!(
                "the answer is larger than the {} bytes one AWS SDK call may read back",
                super::aws_api::MAX_RESPONSE_BYTES
            ),
        ),
        Refusal::NotMade(detail) => refusal_answer("not_made", detail),
    }
}

/// What the guest reads back from one request it sent itself: the service's own status,
/// headers and body, or the reason no request was made.
fn request_answer(answered: &Result<super::guest_http::Answered, Refusal>) -> String {
    use ::base64::Engine;

    match answered {
        Ok(answer) => ::serde_json::json!({
            "status": answer.status,
            "reason": ::http::StatusCode::from_u16(answer.status)
                .ok()
                .and_then(|status| status.canonical_reason()),
            "headers": answer.headers,
            "body": ::base64::engine::general_purpose::STANDARD.encode(&answer.body),
        })
        .to_string(),
        Err(refused) => refusal_answer_for(refused),
    }
}

/// The answer for a call that was not made, naming which refusal it was.
fn refusal_answer(refused: &str, detail: &str) -> String {
    ::serde_json::json!({ "refused": refused, "detail": detail }).to_string()
}

/// The names Python's `os` looks for on the module it is built on, every one of which
/// lands on the HOST where a real runtime answers it: a file, a directory, a descriptor, a
/// process, a user or the host clock's own accounting.
///
/// Each one EXISTS on this emulator's `posix` (`os` builds itself out of the names it
/// finds, so an absent one changes what `os` offers) and each one RAISES when it is
/// called: a Lambda function in this emulator has no host filesystem and no host process
/// to reach. Nothing here is a stub that pretends to have done the work.
const REFUSED_POSIX_CALLS: &[&str] = &[
    "access", "chdir", "chmod", "chown", "close", "closerange", "confstr", "cpu_count",
    "device_encoding", "dup", "dup2",
    "execv", "execve", "fchdir", "fchmod", "fchown", "fdatasync", "fdopen", "fork", "forkpty",
    "fstat", "fstatvfs", "fsync", "ftruncate", "get_blocking", "get_inheritable",
    "get_terminal_size", "getegid", "geteuid", "getgid", "getgrouplist", "getgroups", "getlogin",
    "getppid", "getpriority", "getsid", "getuid", "isatty", "kill", "killpg", "lchown", "link",
    "listdir", "lseek", "lstat", "major", "makedev", "minor", "mkdir", "mkfifo", "mknod", "nice",
    "open", "openpty", "pipe", "popen", "pread", "putenv_host", "pwrite", "read", "readlink",
    "readv", "remove", "rename", "replace", "rmdir", "scandir", "sched_yield", "sendfile",
    "set_blocking", "set_inheritable", "setegid", "seteuid", "setgid", "setgroups", "setpgid",
    "setpgrp", "setpriority", "setregid", "setreuid", "setsid", "setuid", "spawnv", "spawnve",
    "pathconf", "fpathconf", "process_cpu_count", "sched_getaffinity", "sched_setaffinity",
    "sysconf", "waitstatus_to_exitcode", "lockf", "getrandom",
    "stat", "statvfs", "symlink", "sync", "system", "tcgetpgrp", "tcsetpgrp", "times", "truncate",
    "ttyname", "umask", "uname", "unlink", "utime", "wait", "wait3", "wait4", "waitpid", "write",
    "writev",
];

/// Put a refusal on `posix` under every name of [`REFUSED_POSIX_CALLS`].
///
/// The module itself is this crate's ([`super::host_posix`]) and carries only the calls a
/// run can be answered from its own memory; these are the names `os` looks for beyond
/// those, so that `os` is built out of a complete surface and every one of them raises.
fn narrow_posix(vm: &VirtualMachine) -> PyResult<()> {
    let system = vm.import("posix", 0)?;
    // `os` builds the sets that say which of its calls take a directory descriptor or
    // follow symbolic links out of this list. It is EMPTY, which is what says that none of
    // them does: there is no host descriptor and no host link to follow.
    let _ = system.set_attr("_have_functions", vm.ctx.new_list(Vec::new()), vm);
    for name in REFUSED_POSIX_CALLS {
        let refusing = vm.new_function(name, |_args: FuncArgs, vm: &VirtualMachine| {
            Err::<PyObjectRef, _>(vm.new_os_error(
                "a Lambda function in this emulator has no host filesystem or process to reach"
                    .to_owned(),
            ))
        });
        let _ = system.set_attr(*name, refusing, vm);
    }
    Ok(())
}

/// Take away every call that would start a HOST thread, on the single `_thread` module
///
/// A real guest thread would run outside the step budget, the stop latch and the memory
/// sample, so none can be started. The build registers no native thread module at all (the
/// engine feature that would is not taken, see this crate's `Cargo.toml`), so the one a
/// guest imports is the engine's own thread-less stub; these refusals are what make that
/// explicit at every name that starts something.
fn refuse_guest_threads(vm: &VirtualMachine) -> PyResult<()> {
    let refusing = vm.new_function(
        "start_new_thread",
        |_args: FuncArgs, vm: &VirtualMachine| {
            Err::<PyObjectRef, _>(vm.new_runtime_error(
                "a Lambda function in this emulator cannot start a thread".to_owned(),
            ))
        },
    );
    if let Ok(threads) = vm.import("_thread", 0) {
        for attribute in [
            "start_new_thread",
            "start_new",
            "start_joinable_thread",
            "interrupt_main",
            "_set_sentinel",
        ] {
            let _ = threads.set_attr(attribute, refusing.clone(), vm);
        }
    }
    Ok(())
}

/// Put one host function on `module` under `name`.
fn set_function<F>(
    vm: &VirtualMachine,
    module: &PyRef<PyModule>,
    name: &'static str,
    function: F,
) where
    F: Fn(FuncArgs, &VirtualMachine) -> PyResult<PyObjectRef> + Send + Sync + 'static,
{
    let callable = vm.new_function(name, function);
    let _ = module.as_object().set_attr(name, callable, vm);
}

/// How an exception of the interpreter ITSELF is reported — never to a caller as a result,
/// only as the internal failure it is.
fn formatted_exception(
    vm: &VirtualMachine,
    exception: &::rustpython_vm::builtins::PyBaseExceptionRef,
) -> String {
    let mut rendered = String::new();
    if vm.write_exception(&mut rendered, exception).is_err() {
        rendered.push_str("an exception that could not be rendered");
    }
    rendered
}

/// The run's budget, as the hook spends it. Shared with the host functions and read by the
/// caller after the run.
#[derive(Debug)]
pub struct Bounds {
    /// When the PHASE the run is in must be over: the initialization of the execution
    /// environment until [`Self::start`], and the handler's own run from there.
    deadline: Mutex<Instant>,
    /// How long the handler may run for.
    wall_clock: Duration,
    /// The clock every deadline of this run is read against, which the HOST hands the run
    /// ([`run_against`]) and no guest code can reach.
    clock: Clock,
    /// When the HANDLER's run must be over, in milliseconds since the epoch, which is what
    /// the handler is told it has left. Set by [`Self::start`].
    deadline_epoch_ms: ::std::sync::atomic::AtomicI64,
    /// How many traced events are left.
    steps: AtomicU64,
    /// How many have been spent, for the memory-sample interval.
    spent: AtomicU64,
    /// How much memory the run may add to this process.
    memory_bytes: u64,
    /// How large the handler's response may be.
    response_bytes: usize,
    /// This thread's faulted bytes when the run's own code was about to run, which is what
    /// the run's own ceiling measures its growth against. Taken by
    /// [`Self::measure_memory_from_here`], which only the HOST calls — no guest call re-takes
    /// it, so no guest can raise the baseline it is judged against.
    faulted_at_start: AtomicU64,
    /// What this PROCESS held resident at that same instant, which is what the process
    /// ceiling measures this run's own growth against. Taken there too, for the same reason.
    resident_at_start: AtomicU64,
    /// Whether those baselines have been taken yet ([`Self::measure_memory_from_here`]),
    /// which is the instant from which a memory reading says something about THIS RUN — and
    /// so the instant from which the host's own watch judges one ([`Self::watch`]). It is
    /// taken before any of the run's own code is read, the module-level code of the
    /// deployment package included, so no stretch of guest work falls outside the watch.
    measured: AtomicBool,
    /// Which bound was breached, once one was. Latched: the answer to a run that spent its
    /// budget is that stop, whatever the guest does afterwards.
    stopped: Mutex<Option<Stop>>,
    /// Whether a bound was breached at all, read without taking the lock.
    breached: AtomicBool,
    /// Whether the INVOKE phase has begun ([`Self::start`]), which is when the handler's
    /// own deadline starts running.
    spending: AtomicBool,
}

impl Bounds {
    /// The bounds of a run with `budget`, in its INIT phase: the deadline is the one the
    /// initialization of the execution environment has, which is not the function's
    /// `Timeout` — the phase has a limit of its own ([`super::INIT_PHASE_LIMIT`]), and AWS
    /// retries an initialization that outlasts it under the configured function timeout, so
    /// what an initialization is given up on after is the two spans together. The handler's
    /// own deadline is taken by [`Self::start`], when the invoke phase begins.
    ///
    /// The MEMORY baselines are taken by [`Self::measure_memory_from_here`], once the
    /// interpreter this emulator builds is standing and before any of the run's own code is
    /// read — by the host, because a baseline a guest could re-take is no ceiling at all.
    fn new(budget: &Budget, clock: Clock) -> Self {
        Self {
            deadline: Mutex::new(
                clock.now() + super::INIT_PHASE_LIMIT.saturating_add(budget.wall_clock),
            ),
            wall_clock: budget.wall_clock,
            clock,
            deadline_epoch_ms: ::std::sync::atomic::AtomicI64::new(0),
            steps: AtomicU64::new(budget.steps),
            spent: AtomicU64::new(0),
            memory_bytes: budget.memory_bytes,
            response_bytes: budget.response_bytes,
            faulted_at_start: AtomicU64::new(faulted_bytes()),
            resident_at_start: AtomicU64::new(resident_bytes()),
            measured: AtomicBool::new(false),
            stopped: Mutex::new(None),
            breached: AtomicBool::new(false),
            spending: AtomicBool::new(false),
        }
    }

    /// Measure this run's memory from HERE: the thread's faulted pages and the process's
    /// resident size as they are now, which is what both ceilings judge the run's own growth
    /// against.
    ///
    /// Called by the HOST only ([`drive`]), once the interpreter is standing and before a
    /// line of the run's own code has been read, so what it measures is the memory the run
    /// itself takes and not the environment this emulator built for it. It is deliberately
    /// not on the module the guest can reach: a run that could re-take its own baseline
    /// could hold the ceiling's worth of memory again after every call.
    fn measure_memory_from_here(&self) {
        self.faulted_at_start
            .store(faulted_bytes(), Ordering::Relaxed);
        self.resident_at_start
            .store(resident_bytes(), Ordering::Relaxed);
        // Last, so the host's own watch ([`Self::watch`]) never judges a run against a
        // baseline that is still the one taken when these bounds were built.
        self.measured.store(true, Ordering::SeqCst);
    }

    /// Begin the INVOKE phase: the handler's deadline runs from HERE, so what the deadline
    /// measures is the handler's own work rather than the execution environment this
    /// emulator built to run it in — the interpreter, the runtime and the import of the
    /// handler's own module, none of which the function's `Timeout` bounds. Answers when
    /// that deadline falls, in milliseconds since the epoch, which is what the handler is
    /// told it has left.
    ///
    /// Called by the HOST, between the two calls it makes into the runtime ([`drive`]), and
    /// reachable from no guest code. It is ONE-SHOT and moves NOTHING but this one deadline
    /// besides: a second call answers the same instant, and no memory baseline, step count
    /// or latched stop is touched.
    fn start(&self) -> i64 {
        if self.spending.swap(true, Ordering::SeqCst) {
            return self.deadline_epoch_ms.load(Ordering::Relaxed);
        }
        *self.deadline.lock().unwrap_or_else(PoisonError::into_inner) =
            self.clock.now() + self.wall_clock;
        // What the handler is TOLD it has left is a host wall-clock instant rather than a
        // bound: the bound is the deadline above, which this run's own clock is read for.
        let epoch_ms = super::epoch_millis(SystemTime::now() + self.wall_clock);
        self.deadline_epoch_ms.store(epoch_ms, Ordering::Relaxed);
        epoch_ms
    }

    /// Sample the run's DEADLINE and the PROCESS's memory from OUTSIDE it, and latch the stop
    /// when either is spent.
    ///
    /// Called by the host's own watch thread ([`watch_run`]) rather than by the interpreter,
    /// so what a run is stopped on is the host's own reading and not the guest's traced events:
    /// a long NATIVE call emits none, and this is what notices a run that has stopped producing
    /// them — whether it is sitting past its deadline or has taken the process past its ceiling
    /// in ONE allocation. The stop is latched here and raised at the next event the interpreter
    /// does deliver, which is the only place an interpreter can be interrupted; the caller's own
    /// wait and abandoned-run accounting ([`super::run_invocation`]) answer the caller
    /// whatever the run goes on doing.
    ///
    /// The run's OWN ceiling stays with the hook ([`Self::step`]): it is this THREAD's faulted
    /// pages, which only the run's own thread can read. The process's resident size is a
    /// reading any thread can take, so from the instant the run's own code can allocate it is
    /// taken here as well — a bound sampled only between traced events is one a single native
    /// allocation would step over, and the module-level code of the deployment package, which
    /// runs in the INIT phase, allocates as freely as the handler does. That instant is when
    /// the baselines were taken ([`Self::measure_memory_from_here`]), which the host does
    /// before it reads a line of the run's own code: before it, while this emulator is
    /// building the interpreter the run will happen in, the memory in flight is the host's own
    /// doing and stopping a run for it would answer a caller that allocated none of it.
    ///
    /// The deadline judged is the one of the PHASE the run is in: a run still initializing
    /// its execution environment is judged against the span that initialization is allowed
    /// ([`super::INIT_PHASE_LIMIT`]), and one whose handler is running against the
    /// function's own `Timeout` — so neither phase is unbounded and neither is charged the
    /// other's span.
    fn watch(&self) {
        if self.stopped().is_some() {
            return;
        }
        if self.clock.now() >= self.deadline() {
            self.breach(Stop::Deadline);
            return;
        }
        if self.measured.load(Ordering::SeqCst) && self.past_process_ceiling() {
            self.breach(Stop::Memory);
        }
    }

    /// When the run must be over.
    fn deadline(&self) -> Instant {
        *self.deadline.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Spend one traced event: `None` to carry on, `Some(stop)` for the bound that is now
    /// breached.
    fn step(&self) -> Option<Stop> {
        if let Some(stop) = self.stopped() {
            return Some(stop);
        }
        if self.steps.fetch_sub(1, Ordering::Relaxed) == 0 {
            return Some(self.breach(Stop::Steps));
        }
        if self.clock.now() >= self.deadline() {
            return Some(self.breach(Stop::Deadline));
        }
        // The run's OWN memory is sampled at EVERY traced event, so there is no sampling
        // interval for an allocating guest to outrun: the thread's faulted pages are a
        // counter this process already keeps. The whole process's resident size is read
        // from the operating system, which is the dearer of the two, so that one is read
        // every [`STEPS_BETWEEN_PROCESS_SAMPLES`] events.
        let spent = self.spent.fetch_add(1, Ordering::Relaxed);
        if self.past_run_ceiling() {
            return Some(self.breach(Stop::Memory));
        }
        if spent.is_multiple_of(STEPS_BETWEEN_PROCESS_SAMPLES) && self.past_process_ceiling() {
            return Some(self.breach(Stop::Memory));
        }
        None
    }

    /// Whether the run has grown past its own ceiling, sampled from OUTSIDE the guest: the
    /// interpreter exposes no allocation seam, so the bound is read from this thread's
    /// faulted pages, against what this thread had faulted when the run's own code was
    /// about to be read ([`Self::measure_memory_from_here`]).
    fn past_run_ceiling(&self) -> bool {
        faulted_bytes().saturating_sub(self.faulted_at_start.load(Ordering::Relaxed))
            > self.memory_bytes
    }

    /// Whether THIS RUN has grown the whole process past [`PROCESS_RESIDENT_CEILING_BYTES`].
    ///
    /// Both halves are needed, and each says something the other does not: the process is past
    /// the ceiling, AND this run is what took it there — it holds more than it held when its
    /// own code was about to be read ([`Self::measure_memory_from_here`]), by more than the
    /// run's own ceiling allows it to add. A process
    /// already past the ceiling when the run began is not this run's doing: stopping the run
    /// for it would answer a caller that allocated none of it and would free none of it
    /// either, while the run's own growth is bounded by the ceiling above whatever the rest of
    /// the process holds.
    ///
    /// Read by the hook and by the host's own watch ([`Self::watch`]), so a run that takes the
    /// process past the ceiling inside one native call is stopped without waiting for the
    /// guest to deliver another traced event.
    fn past_process_ceiling(&self) -> bool {
        let held = resident_bytes();
        if held <= PROCESS_RESIDENT_CEILING_BYTES {
            return false;
        }
        held.saturating_sub(self.resident_at_start.load(Ordering::Relaxed)) > self.memory_bytes
    }

    /// Latch `stop` as the run's answer and report it.
    fn breach(&self, stop: Stop) -> Stop {
        let mut latched = self.stopped.lock().unwrap_or_else(PoisonError::into_inner);
        let already = *latched.get_or_insert(stop);
        self.breached.store(true, Ordering::Relaxed);
        already
    }

    /// The bound this run breached, if it breached one.
    #[must_use]
    pub fn stopped(&self) -> Option<Stop> {
        if !self.breached.load(Ordering::Relaxed) {
            return None;
        }
        *self.stopped.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// How long the run has left.
    fn remaining(&self) -> Duration {
        self.deadline().saturating_duration_since(self.clock.now())
    }
}

/// How many bytes of memory THIS thread has faulted in, as the host reports it.
///
/// Linux reports faults per thread. macOS has no per-thread fault count, so there the figure
/// covers the whole process: a run then also counts faults made by other runs at the same
/// time, which can stop a run earlier than its own ceiling would.
fn faulted_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    let scope = ::nix::sys::resource::UsageWho::RUSAGE_THREAD;
    #[cfg(not(target_os = "linux"))]
    let scope = ::nix::sys::resource::UsageWho::RUSAGE_SELF;
    match ::nix::sys::resource::getrusage(scope) {
        Ok(usage) => {
            let pages = usage.minor_page_faults().saturating_add(usage.major_page_faults());
            u64::try_from(pages).unwrap_or(0).saturating_mul(page_bytes())
        }
        Err(_) => 0,
    }
}

/// How many bytes this PROCESS holds resident, as the host reports it.
fn resident_bytes() -> u64 {
    match ::memory_stats::memory_stats() {
        Some(stats) => u64::try_from(stats.physical_mem).unwrap_or(0),
        None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A budget small enough to exhaust in a test and large enough for any handler one
    /// runs.
    ///
    /// Its memory ceiling is out of reach ([`CEILING_OUT_OF_REACH`]), because a row that is
    /// not about the ceiling must not be answered by it: the ceiling is read from the whole
    /// PROCESS as well as from the run, and this binary's own growth belongs to no row. A row
    /// about the ceiling gives itself one it can reach.
    fn budget() -> Budget {
        Budget {
            wall_clock: Duration::from_secs(5),
            steps: 2_000_000,
            memory_bytes: CEILING_OUT_OF_REACH,
            log_bytes: 16 * 1024,
            response_bytes: 64 * 1024,
            recursion_limit: 2_000,
        }
    }

    /// How long this module waits for a run's own thread to end before it reports the run as
    /// one that did not end at all.
    ///
    /// NOT a bound on the run: every bound a run has is the host's own, read inside the run
    /// ([`Bounds`]). This is the backstop that keeps a run which will not end from wedging
    /// the suite — the host gives such a run up and answers its caller anyway
    /// ([`super::run_invocation`]), and so does this module. It is far above any span a
    /// row's own bound takes, so no row is ever answered by it.
    const RUN_ANSWER_CEILING: Duration = Duration::from_secs(240);

    /// How far ONE reading drives the clock a row about the DEADLINE is judged against.
    ///
    /// A deadline read against a driven clock ([`Clock::driven`]) falls after a NUMBER of
    /// readings — the run's own traced events, and the host watch's samples of it — rather
    /// than after a span of this host's time, so which bound answers such a row is the same
    /// on an idle host and on a loaded one. It is small against the span it has to drive the
    /// clock across, so the phase a row is about is reached long before the readings taken
    /// getting there have driven the deadline past it.
    const DRIVEN_TICK: Duration = Duration::from_micros(10);

    /// A memory ceiling a row about any OTHER bound cannot reach: the largest `MemorySize` a
    /// function may have (`smithy_query(query="shape", shape="MemorySize")`:
    /// `@range(min: 128, max: 32768)`), in bytes.
    ///
    /// The ceiling is read from the whole PROCESS as well as from the run
    /// ([`Bounds::past_process_ceiling`]): a run that allocated nothing of its own is stopped
    /// for memory once this process has grown past its ceiling by more than the run's own
    /// ceiling allows it to add — and in a test binary running many suites at once, that
    /// growth is other suites' and no row's own. A row that names another bound is therefore
    /// given the largest ceiling a function may have, which neither half of the memory bound
    /// can reach, so the bound the row names is the only one that can answer it.
    const CEILING_OUT_OF_REACH: u64 = 32_768 * 1024 * 1024;

    /// Run `source` as the module `lambda_function` of a one-file package, on a thread with
    /// the stack a run is given — the same way [`super::run_invocation`] runs one.
    fn ran(source: &str, handler: &str, event: Option<&str>) -> Finished {
        ran_with(source, handler, event, &BTreeMap::new(), budget())
    }

    /// [`ran`] with an environment and a budget of its own, against this host's own clock.
    fn ran_with(
        source: &str,
        handler: &str,
        event: Option<&str>,
        environment: &BTreeMap<String, String>,
        budget: Budget,
    ) -> Finished {
        ran_against(
            source,
            handler,
            event,
            environment,
            budget,
            &Clock::host(),
            "a run against this host's own clock",
        )
    }

    /// [`ran_with`] against `clock`, which is what leaves ONE bound able to answer the run:
    /// a row that names a bound other than the deadline is judged against a clock that does
    /// not advance, and a row that names the deadline against one the run's own readings
    /// drive.
    ///
    /// The run's answer is waited for rather than joined on, so a run that will not end
    /// fails the row that asked for it ([`RUN_ANSWER_CEILING`]) instead of holding the suite.
    fn ran_against(
        source: &str,
        handler: &str,
        event: Option<&str>,
        environment: &BTreeMap<String, String>,
        budget: Budget,
        clock: &Clock,
        case: &str,
    ) -> Finished {
        let modules = vec![PackageModule {
            path: "lambda_function.py".to_string(),
            source: source.to_string(),
        }];
        let handler = handler.to_string();
        let event = event.map(str::to_string);
        let environment = environment.clone();
        let clock = clock.clone();
        let (answered, answer) = ::std::sync::mpsc::channel();
        ::std::thread::Builder::new()
            .stack_size(crate::state::execution::RUN_STACK_BYTES)
            .spawn(move || {
                let finished = run_against(
                    &Run {
                        modules: &modules,
                        handler: &handler,
                        environment: &environment,
                        event: event.as_deref(),
                        request_id: "11111111-2222-3333-4444-555555555555",
                        invoked_function_arn:
                            "arn:aws:lambda:us-west-2:000000000000:function:probe",
                        client_context: None,
                        tenant_id: None,
                        budget,
                        // These runs are the interpreter's own containment and bounds: no
                        // emulator is standing behind them, so there is no SDK seam either.
                        aws: None,
                    },
                    &clock,
                );
                let _ = answered.send(finished);
            })
            .expect("a run thread");
        answer.recv_timeout(RUN_ANSWER_CEILING).unwrap_or_else(|_| {
            panic!("case {case}: the run did not end, so no bound of it ended it")
        })
    }

    /// The JSON the handler returned, or a panic naming what it came to instead.
    fn returned(finished: &Finished, case: &str) -> ::serde_json::Value {
        match &finished.completion {
            Completion::Returned(payload) => ::serde_json::from_str(payload)
                .unwrap_or_else(|_| panic!("case {case}: expected JSON, got {payload:?}")),
            other => panic!("case {case}: expected a returned value, got {other:?}"),
        }
    }

    /// The handler's own value is what comes back, for one row per JSON type a handler can
    /// return — so the answer is the run's, not a shape this emulator fills in.
    #[test]
    fn the_answer_is_what_the_handler_returned() {
        let cases: &[(&str, &str, &str)] = &[
            ("a bool member", "{'ok': True}", r#"{"ok":true}"#),
            ("a computed number", "{'n': sum(range(10))}", r#"{"n":45}"#),
            ("a string", "'hello'", r#""hello""#),
            ("a list", "[1, 2, 3]", "[1,2,3]"),
            ("null", "None", "null"),
            (
                "the event echoed back",
                "{'event': event}",
                r#"{"event":{"k":"v"}}"#,
            ),
            (
                "a value from the standard library",
                "{'encoded': __import__('base64').b64encode(b'ab').decode()}",
                r#"{"encoded":"YWI="}"#,
            ),
            (
                "a value from json itself",
                "{'parsed': __import__('json').loads('[1, 2]')}",
                r#"{"parsed":[1,2]}"#,
            ),
        ];
        for (case, expression, expected) in cases {
            let source = format!("def handler(event, context):\n    return {expression}\n");
            let finished = ran(&source, "lambda_function.handler", Some(r#"{"k": "v"}"#));
            let actual = returned(&finished, case);
            let wanted: ::serde_json::Value =
                ::serde_json::from_str(expected).expect("the expected JSON");
            assert_eq!(
                actual, wanted,
                "case {case}: expected {wanted}, got {actual} (log {:?})",
                finished.log
            );
        }
    }

    /// The context object reports the invocation, out of the environment the run was given.
    #[test]
    fn the_context_reports_the_invocation() {
        let mut environment = BTreeMap::new();
        environment.insert("AWS_LAMBDA_FUNCTION_NAME".to_string(), "probe".to_string());
        environment.insert("AWS_LAMBDA_FUNCTION_VERSION".to_string(), "7".to_string());
        environment.insert(
            "AWS_LAMBDA_FUNCTION_MEMORY_SIZE".to_string(),
            "512".to_string(),
        );
        environment.insert("GREETING".to_string(), "hello".to_string());
        let source = "import os\n\
                      def handler(event, context):\n    \
                          return {\n        \
                              'name': context.function_name,\n        \
                              'version': context.function_version,\n        \
                              'memory': context.memory_limit_in_mb,\n        \
                              'arn': context.invoked_function_arn,\n        \
                              'request': context.aws_request_id,\n        \
                              'left': context.get_remaining_time_in_millis() > 0,\n        \
                              'greeting': os.environ['GREETING'],\n        \
                              'cwd': os.getcwd(),\n    \
                          }\n";
        let finished = ran_with(
            source,
            "lambda_function.handler",
            None,
            &environment,
            budget(),
        );
        let answered = returned(&finished, "the context");
        let cases: &[(&str, ::serde_json::Value)] = &[
            ("name", ::serde_json::json!("probe")),
            ("version", ::serde_json::json!("7")),
            ("memory", ::serde_json::json!("512")),
            (
                "arn",
                ::serde_json::json!("arn:aws:lambda:us-west-2:000000000000:function:probe"),
            ),
            (
                "request",
                ::serde_json::json!("11111111-2222-3333-4444-555555555555"),
            ),
            ("left", ::serde_json::json!(true)),
            ("greeting", ::serde_json::json!("hello")),
            ("cwd", ::serde_json::json!("/var/task")),
        ];
        for (member, expected) in cases {
            let actual = answered.get(*member).cloned().unwrap_or_default();
            assert_eq!(
                actual, *expected,
                "case context.{member}: expected {expected}, got {actual} (log {:?})",
                finished.log
            );
        }
    }

    /// What the handler wrote to its output streams is the run's log, and nothing of the
    /// host's.
    #[test]
    fn the_log_is_what_the_handler_wrote() {
        let source = "import sys\n\
                      def handler(event, context):\n    \
                          print('to stdout')\n    \
                          sys.stderr.write('to stderr\\n')\n    \
                          return 1\n";
        let finished = ran(source, "lambda_function.handler", None);
        for expected in ["to stdout", "to stderr"] {
            assert!(
                finished.log.contains(expected),
                "case {expected:?}: expected it in the log, got {:?}",
                finished.log
            );
        }
    }

    /// An exception the handler raised is reported as the exception it is, with the
    /// handler's own stack.
    #[test]
    fn an_exception_the_handler_raised_is_reported_as_itself() {
        let cases: &[(&str, &str, &str)] = &[
            ("raise ValueError('boom')", "ValueError", "boom"),
            ("raise KeyError('k')", "KeyError", "'k'"),
            ("return 1 / 0", "ZeroDivisionError", "division by zero"),
            (
                "return {'set': set()}",
                "Runtime.MarshalError",
                "Unable to marshal response",
            ),
        ];
        for (statement, error_type, message) in cases {
            let source = format!("def handler(event, context):\n    {statement}\n");
            let finished = ran(&source, "lambda_function.handler", None);
            match &finished.completion {
                Completion::Raised {
                    error_type: actual_type,
                    error_message,
                    stack_trace,
                } => {
                    assert_eq!(
                        actual_type, error_type,
                        "case {statement:?}: expected the type {error_type}, got {actual_type}"
                    );
                    assert!(
                        error_message.contains(message),
                        "case {statement:?}: expected {message:?} in the message, got \
                         {error_message:?}"
                    );
                    if !error_type.starts_with("Runtime.") {
                        assert!(
                            stack_trace.iter().any(|frame| frame.contains("/var/task/")),
                            "case {statement:?}: expected the handler's own frame, got \
                             {stack_trace:?}"
                        );
                    }
                }
                other => panic!("case {statement:?}: expected a raise, got {other:?}"),
            }
        }
    }

    /// A handler that cannot be resolved is the runtime failure for it, named as the real
    /// runtime names it — one row per way a handler fails to resolve.
    #[test]
    fn a_handler_that_does_not_resolve_is_reported_as_the_runtime_failure() {
        let source = "def handler(event, context):\n    return 1\n";
        let cases: &[(&str, &str)] = &[
            ("lambda_function.absent", "Runtime.HandlerNotFound"),
            ("absent_module.handler", "Runtime.ImportModuleError"),
            ("handler", "Runtime.MalformedHandlerName"),
        ];
        for (handler, expected) in cases {
            let finished = ran(source, handler, None);
            match &finished.completion {
                Completion::Raised { error_type, .. } => assert_eq!(
                    error_type, expected,
                    "case {handler:?}: expected {expected}, got {error_type}"
                ),
                other => panic!("case {handler:?}: expected a raise, got {other:?}"),
            }
        }
    }

    /// Every bound is the HOST's: one row per bound, each exhausted from guest code, and
    /// each ends the run with that bound rather than with an answer.
    ///
    /// A row names ONE bound, so the row is given a run only that bound can answer: the
    /// CLOCK the run's deadline is read against is the row's own
    /// ([`super::super::run_clock`]), and a row about a bound that is not a span of time is
    /// judged against a clock that does not advance — which is what keeps a host slow enough
    /// to spend the deadline from answering the row with the deadline instead of with the
    /// bound the row is about. The MEMORY ceiling is put out of reach of every row but the
    /// one about it for the same reason ([`CEILING_OUT_OF_REACH`]). The last two rows are the
    /// same guest code under two different runs, so what differs between their answers is the
    /// bound each run left reachable and nothing of the host.
    #[test]
    fn a_run_that_breaches_a_bound_is_stopped_by_it() {
        /// One row: the bound it is about, the handler body that reaches it, and the run
        /// that leaves that bound the only one able to answer.
        struct BoundCase<'a> {
            /// What the row is about.
            case: &'a str,
            /// The handler's body, which reaches the bound below.
            body: &'a str,
            /// The bound that has to end the run.
            expected: Stop,
            /// The clock the run's deadline is read against.
            clock: Clock,
            /// How many traced events the run may spend: a row about the DEADLINE has to
            /// reach it before the step budget answers instead.
            steps: u64,
            /// How long the handler may run for, in that clock's own time.
            wall_clock: Duration,
            /// How much memory the run may add to this process.
            memory_bytes: u64,
        }
        // Enough events for a row about another bound to reach it, and far fewer than a row
        // about the deadline drives its clock across.
        let few_steps = 50_000;
        let many_steps = 2_000_000_000;
        // Small enough that an allocating handler passes it inside the step budget, and far
        // above what building the interpreter itself faults in.
        let reachable_ceiling = 64 * 1024 * 1024;
        let cases = vec![
            BoundCase {
                case: "a loop with no exit",
                body: "while True:\n        pass\n",
                expected: Stop::Steps,
                clock: Clock::still(),
                steps: few_steps,
                wall_clock: budget().wall_clock,
                memory_bytes: CEILING_OUT_OF_REACH,
            },
            BoundCase {
                case: "unbounded recursion",
                body: "def deep(n):\n        return deep(n + 1)\n    deep(0)\n",
                expected: Stop::Steps,
                clock: Clock::still(),
                steps: few_steps,
                wall_clock: budget().wall_clock,
                memory_bytes: CEILING_OUT_OF_REACH,
            },
            BoundCase {
                case: "a response larger than one may be",
                body: "return 'x' * 200000\n",
                expected: Stop::Response,
                clock: Clock::still(),
                steps: few_steps,
                wall_clock: budget().wall_clock,
                memory_bytes: CEILING_OUT_OF_REACH,
            },
            BoundCase {
                case: "an allocating loop",
                body: "kept = []\n    \
                       while True:\n        \
                           kept.append(bytearray(1024 * 1024))\n",
                expected: Stop::Memory,
                clock: Clock::still(),
                steps: few_steps,
                wall_clock: budget().wall_clock,
                memory_bytes: reachable_ceiling,
            },
            BoundCase {
                case: "a loop with no exit, with events enough to drive its clock past the \
                       deadline",
                body: "while True:\n        pass\n",
                expected: Stop::Deadline,
                clock: Clock::driven(DRIVEN_TICK),
                steps: many_steps,
                wall_clock: Duration::from_secs(1),
                memory_bytes: CEILING_OUT_OF_REACH,
            },
        ];
        for case in &cases {
            let source = format!("def handler(event, context):\n    {}", case.body);
            let mut tight = budget();
            tight.steps = case.steps;
            tight.wall_clock = case.wall_clock;
            tight.memory_bytes = case.memory_bytes;
            let finished = ran_against(
                &source,
                "lambda_function.handler",
                None,
                &BTreeMap::new(),
                tight,
                &case.clock,
                case.case,
            );
            let stopped = match &finished.completion {
                Completion::Stopped(stop) => Some(*stop),
                // Deep recursion may end at the interpreter's own clamped limit first,
                // which is also the host's bound doing its job.
                Completion::Raised { error_type, .. } if error_type == "RecursionError" => None,
                other => panic!("case {}: expected a stop, got {other:?}", case.case),
            };
            if let Some(stopped) = stopped {
                assert_eq!(
                    stopped, case.expected,
                    "case {}: expected {:?}, got {stopped:?}",
                    case.case, case.expected
                );
            }
        }
    }

    /// The step budget cannot be escaped by catching what the hook raises: the stop is
    /// LATCHED, so the run's answer is the stop whatever the guest returns afterwards.
    ///
    /// The run is judged against a clock that does not advance, so the stop the guest
    /// catches is the step budget whatever this host's own clock did meanwhile.
    #[test]
    fn a_guest_that_catches_the_stop_is_still_stopped() {
        let source = "def handler(event, context):\n    \
                          try:\n        \
                              while True:\n            \
                                  pass\n    \
                          except BaseException:\n        \
                              return {'escaped': True}\n";
        let mut tight = budget();
        tight.steps = 20_000;
        let case = "a guest that catches the stop it was given";
        let finished = ran_against(
            source,
            "lambda_function.handler",
            None,
            &BTreeMap::new(),
            tight,
            &Clock::still(),
            case,
        );
        assert!(
            matches!(finished.completion, Completion::Stopped(Stop::Steps)),
            "case {case}: expected the latched step stop to win, got {:?}",
            finished.completion
        );
    }

    /// The deadline is the host's too: a sleep the guest asks for never outlasts the run.
    #[test]
    fn a_sleep_past_the_deadline_does_not_outlast_the_run() {
        let source = "import time\n\
                      def handler(event, context):\n    \
                          time.sleep(600)\n    \
                          return {'slept': True}\n";
        let mut tight = budget();
        tight.wall_clock = Duration::from_secs(2);
        let started = Instant::now();
        let finished = ran_with(
            source,
            "lambda_function.handler",
            None,
            &BTreeMap::new(),
            tight,
        );
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "expected the clamped sleep to end inside the run, took {:?}",
            started.elapsed()
        );
        assert!(
            !matches!(finished.completion, Completion::Internal(_)),
            "expected the run to end on its own bound, got {:?}",
            finished.completion
        );
    }

    /// Containment, attempted FROM guest code: one row per host surface a handler must not
    /// reach. Each row is the expression, and each has to fail.
    #[test]
    fn no_host_surface_is_reachable_from_guest_code() {
        let cases: &[(&str, &str)] = &[
            ("a subprocess", "__import__('subprocess')"),
            ("a shell command", "__import__('os').system('echo hi')"),
            ("a process pipe", "__import__('os').popen('echo hi')"),
            ("a host file descriptor", "__import__('os').open('/etc/hostname', 0)"),
            ("a host file removal", "__import__('os').remove('/etc/hostname')"),
            ("a fork", "__import__('os').fork()"),
            // A socket EXISTS — a traced handler writes its segment documents to the X-Ray
            // daemon through one ([`super::xray_daemon`]) — and the containment is that it
            // reaches that daemon and nothing else: a connection to a host, and a datagram
            // addressed at one, each have no route.
            (
                "a connection to a host",
                "__import__('socket').socket().connect(('example.com', 80))",
            ),
            (
                "a datagram to a host",
                "__import__('socket').socket(2, 2).sendto(b'x', ('example.com', 80))",
            ),
            (
                "a listening socket",
                "__import__('socket').socket().bind(('0.0.0.0', 8080)) or \
                 __import__('socket').socket().listen(1)",
            ),
            (
                "a host name lookup",
                "__import__('socket').gethostbyname('example.com')",
            ),
            ("the host's posix", "__import__('posix').open('/etc/hostname', 0)"),
            ("a host file", "open('/etc/hostname').read()"),
            ("a host directory", "__import__('os').listdir('/')"),
            ("a thread", "__import__('_thread').start_new_thread(print, ())"),
            ("a memory map", "__import__('mmap')"),
            ("a host library", "__import__('ctypes')"),
            ("the host's environment", "__import__('os').environ['PATH']"),
            ("a re-import of a dropped module", "__import__('sys').modules.pop('posix', None) or __import__('posix').open('/etc/hostname', 0)"),
            ("the trace hook", "__import__('sys').settrace(None)"),
            ("the recursion limit", "__import__('sys').setrecursionlimit(1000000)"),
            ("the process", "__import__('os')._exit(0)"),
        ];
        for (case, expression) in cases {
            let source = format!("def handler(event, context):\n    return {expression}\n");
            let finished = ran(&source, "lambda_function.handler", None);
            match &finished.completion {
                Completion::Raised { .. } | Completion::Stopped(_) => {}
                other => panic!(
                    "case {case}: expected guest code reaching {case} to fail, got {other:?} \
                     (log {:?})",
                    finished.log
                ),
            }
        }
    }

    /// Containment of the TYPE behind a host file, attempted the way a Python-level denylist
    /// is sidestepped: not by the name `_io.FileIO`, which is replaced, but by walking the
    /// interpreter's type graph from `object` to the type itself. Constructing it opens
    /// nothing, because the refusal is on the type's own construction.
    #[test]
    fn the_type_behind_a_host_file_refuses_however_it_is_reached() {
        let source = "def handler(event, context):\n    \
                          def found(cls):\n        \
                              for sub in cls.__subclasses__():\n            \
                                  if sub.__name__ == 'FileIO':\n                \
                                      return sub\n            \
                                  deeper = found(sub)\n            \
                                  if deeper is not None:\n                \
                                      return deeper\n        \
                              return None\n    \
                          reached = found(object)\n    \
                          if reached is None:\n        \
                              return {'reached': False}\n    \
                          try:\n        \
                              held = reached('/etc/hostname').read()\n    \
                          except BaseException as error:\n        \
                              return {'reached': True, 'refused': type(error).__name__}\n    \
                          return {'reached': True, 'opened': True, 'held': len(held)}\n";
        let finished = ran(source, "lambda_function.handler", None);
        match &finished.completion {
            // Reaching the type at all is allowed; opening a host file with it is not.
            Completion::Raised { .. } | Completion::Stopped(_) => {}
            Completion::Returned(payload) => assert!(
                !payload.contains("\"opened\"")
                    && (payload.contains("\"refused\"") || payload.contains("\"reached\": false")
                        || payload.contains("\"reached\":false")),
                "expected no host file to be opened through the type graph, got {payload}"
            ),
            other => panic!("expected the type behind a host file to refuse, got {other:?}"),
        }
    }

    /// The step, deadline and memory bounds are the HOST's: a handler that rebinds every name
    /// the host module carries — including one it may expect the hook to be behind — and then
    /// spins is still stopped, because the hook the interpreter calls is not reached through
    /// any name the guest can write.
    #[test]
    fn a_guest_that_rebinds_the_host_module_is_still_bounded() {
        let source = "def handler(event, context):\n    \
                          host = __import__('_lambda_host')\n    \
                          for name in dir(host) + ['step']:\n        \
                              try:\n            \
                                  setattr(host, name, lambda *a, **k: None)\n        \
                              except Exception:\n            \
                                  pass\n    \
                          while True:\n        \
                              pass\n";
        let mut tight = budget();
        // A small STEP budget, so the stop the guest cannot reach arrives at once: with the
        // hook switched off nothing would arrive at all. The run is judged against a clock
        // that does not advance, so that budget is the only bound that can answer it and
        // nothing of this host's own speed decides what the row came to.
        tight.steps = 20_000;
        let case = "a guest that rebound every name the host module carries";
        let finished = ran_against(
            source,
            "lambda_function.handler",
            None,
            &BTreeMap::new(),
            tight,
            &Clock::still(),
            case,
        );
        assert!(
            matches!(finished.completion, Completion::Stopped(Stop::Steps)),
            "case {case}: expected the run to be stopped by the step budget the guest cannot \
             reach, got {:?}",
            finished.completion
        );
    }

    /// Where one PHASE of a run ends and the next begins is the HOST's decision: the host
    /// module carries no name for it, so no guest can start its own invoke phase — and a
    /// guest that goes looking for one over every name that module does carry is still
    /// stopped by the bounds the host set.
    ///
    /// The rows differ along WHICH bound such a guest would move if it could: the deadline,
    /// looked for over and over from a handler that never returns, and the memory baseline,
    /// looked for while the handler's module is being imported between two allocations that
    /// TOGETHER pass the run's ceiling — a run that must still be stopped, because the
    /// ceiling is measured from the one baseline the host took.
    #[test]
    fn no_guest_code_can_begin_its_own_invoke_phase() {
        // The names the host module really carries, read from inside a run: a surface this
        // small is the whole of what a guest can call, and none of these is a phase or a
        // bound. A name added to it has to be added here too, which is the point.
        let listed = ran(
            "def handler(event, context):\n    \
                 host = __import__('_lambda_host')\n    \
                 return sorted(n for n in dir(host) if not n.startswith('_'))\n",
            "lambda_function.handler",
            None,
        );
        let reachable = returned(&listed, "the host surface a run can call");
        let expected = ::serde_json::json!([
            "aws",
            "db_auth_token",
            "log",
            "pagination",
            "presign",
            "request",
            "sleep"
        ]);
        assert_eq!(
            reachable, expected,
            "expected the host module to carry {expected} and nothing that begins a phase \
             or moves a bound, got {reachable}"
        );

        /// How many megabytes each half of the memory row below allocates.
        const ALLOCATIONS: usize = 40;
        // Every name the host module carries, called by the guest in the hope that one of
        // them starts its invoke phase over.
        let hunts = "_host = __import__('_lambda_host')\n\
             def _hunt():\n    \
                 for name in dir(_host):\n        \
                     try:\n            \
                         getattr(_host, name)()\n        \
                     except BaseException:\n            \
                         pass\n";
        let hunts_the_deadline = format!(
            "{hunts}\
             _hunt()\n\
             def handler(event, context):\n    \
                 while True:\n        \
                     _hunt()\n"
        );
        let hunts_the_baseline = format!(
            "{hunts}\
             def megabyte():\n    \
                 return bytearray(b'x' * (1024 * 1024))\n\
             KEPT = [megabyte() for _ in range({ALLOCATIONS})]\n\
             _hunt()\n\
             def handler(event, context):\n    \
                 KEPT.extend(megabyte() for _ in range({ALLOCATIONS}))\n    \
                 return {{'held': len(KEPT)}}\n"
        );
        /// One row: what it is about, the package's module source, the bound that has to end
        /// the run, the clock the run is judged against, the span the handler is given in
        /// that clock's own time, and the ceiling the run may add to this process.
        type PhaseCase<'a> = (&'a str, &'a str, Stop, Clock, Duration, u64);
        let cases: &[PhaseCase<'_>] = &[
            (
                "the deadline, hunted for from inside a handler that never returns",
                hunts_the_deadline.as_str(),
                Stop::Deadline,
                Clock::driven(DRIVEN_TICK),
                Duration::from_secs(1),
                CEILING_OUT_OF_REACH,
            ),
            (
                "the memory baseline, hunted for while the module is being imported",
                hunts_the_baseline.as_str(),
                Stop::Memory,
                Clock::still(),
                budget().wall_clock,
                // Under what the two allocations hold together and over what either holds
                // on its own, so this row is stopped only if the baseline stayed where the
                // host took it.
                64 * 1024 * 1024,
            ),
        ];
        for (case, source, expected, clock, wall_clock, memory_bytes) in cases {
            let mut tight = budget();
            // The bound each row names has to be the only one that can answer it, and WHICH
            // clock the run's deadline is read against is what settles that: the row about
            // the DEADLINE is judged against a clock its own traced events drive, so the
            // deadline falls after a number of events rather than after a span of this
            // host's time, and the row about the BASELINE against a clock that does not
            // advance, so no deadline falls for it at all however slowly this host faults
            // its allocations in. Both are given far more steps than either spends, so the
            // step budget answers neither, and the row about the deadline is given a ceiling
            // out of reach, so the memory bound answers it no more than the clock does.
            tight.wall_clock = *wall_clock;
            tight.steps = 2_000_000_000;
            tight.memory_bytes = *memory_bytes;
            let finished = ran_against(
                source,
                "lambda_function.handler",
                None,
                &BTreeMap::new(),
                tight,
                clock,
                case,
            );
            match &finished.completion {
                Completion::Stopped(stopped) => assert_eq!(
                    stopped, expected,
                    "case {case}: expected the run to be stopped by {expected:?}, got \
                     {stopped:?}"
                ),
                other => panic!(
                    "case {case}: expected the run to be stopped by {expected:?}, got {other:?}"
                ),
            }
        }
    }

    /// A guest `atexit` callback runs during the interpreter's finalization, which is after
    /// the handler returned — and it is bounded like every other guest frame: the hook is
    /// still installed and still answers itself as the local tracer, so a callback that spins
    /// is stopped on one of the run's own bounds rather than wedging the thread.
    #[test]
    fn a_guest_atexit_callback_is_bounded_like_the_handler() {
        let source = "import atexit\n\
                      def spin():\n    \
                          while True:\n        \
                              pass\n\
                      def handler(event, context):\n    \
                          atexit.register(spin)\n    \
                          return {'registered': True}\n";
        let mut tight = budget();
        // The callback is stopped on a budget, so this row is about WHETHER it is bounded and
        // not about how much it is given: a small step budget keeps the spin short, and a
        // clock that does not advance leaves that budget the only bound that can answer the
        // run — the stop here is the callback's own spending and not this host's speed.
        tight.steps = 20_000;
        let case = "an atexit callback that spins during the interpreter's finalization";
        let finished = ran_against(
            source,
            "lambda_function.handler",
            None,
            &BTreeMap::new(),
            tight,
            &Clock::still(),
            case,
        );
        assert!(
            matches!(finished.completion, Completion::Stopped(Stop::Steps)),
            "case {case}: expected the spinning callback to spend the run's step budget, got \
             {:?}",
            finished.completion
        );
    }

    /// The ENVIRONMENT a run sees is the invocation's own and nothing of this process's:
    /// read from inside the run, which is where it matters.
    #[test]
    fn the_environment_a_run_sees_holds_nothing_of_the_host() {
        let mut environment = BTreeMap::new();
        environment.insert("SECRET".to_string(), "shhh".to_string());
        let source = "import os\n\
                      def handler(event, context):\n    \
                          return {'own': os.environ.get('SECRET'), \
                                  'names': sorted(os.environ.keys())}\n";
        let finished = ran_with(
            source,
            "lambda_function.handler",
            None,
            &environment,
            budget(),
        );
        let answered = returned(&finished, "the environment");
        assert_eq!(
            answered.get("own").and_then(::serde_json::Value::as_str),
            Some("shhh"),
            "expected the invocation's own variable, got {answered}"
        );
        let names: Vec<&str> = answered
            .get("names")
            .and_then(::serde_json::Value::as_array)
            .map(|names| {
                names
                    .iter()
                    .filter_map(::serde_json::Value::as_str)
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(
            names,
            vec!["SECRET"],
            "expected only the invocation's own variables, got {names:?}"
        );
    }

    /// What this runtime presents of the STANDARD LIBRARY, read from inside a run: one row
    /// per module a handler imports that a module table missing a name of `os` could not
    /// import at all, plus the `stat_result` type `os` re-exports.
    ///
    /// The modules are the ones the standard library itself cannot be imported without:
    /// `shutil` (and `tempfile`, `zipfile`, `tarfile`, which import it) reads
    /// `os.stat_result`, `glob` (and `pathlib`, which imports it) reads `os.O_RDONLY`,
    /// `random` (and `secrets`, `statistics`) registers an at-fork hook, and `http.client`
    /// and `urllib.request` are built on `socket`.
    #[test]
    fn the_standard_library_a_run_presents_is_a_complete_one() {
        let modules = [
            "shutil",
            "zipfile",
            "tempfile",
            "tarfile",
            "pathlib",
            "glob",
            "random",
            "secrets",
            "statistics",
            "socket",
            "http.client",
            "urllib.request",
            "importlib.metadata",
        ];
        let source = "def handler(event, context):\n    \
                          read = {}\n    \
                          for name in event['modules']:\n        \
                              try:\n            \
                                  __import__(name)\n            \
                                  read[name] = 'ok'\n        \
                              except BaseException as error:\n            \
                                  read[name] = repr(error)\n    \
                          return read\n";
        let event = ::serde_json::json!({ "modules": modules }).to_string();
        // Importing this much of the standard library is more work than the small budget
        // above allows, and is work a real invocation's own budget covers
        // ([`crate::state::execution`]), so this row is given a run's own.
        let mut whole = budget();
        whole.wall_clock = Duration::from_secs(120);
        whole.steps = 20_000_000;
        let finished = ran_with(
            source,
            "lambda_function.handler",
            Some(&event),
            &BTreeMap::new(),
            whole,
        );
        let answered = returned(&finished, "the standard library");
        for module in modules {
            assert_eq!(
                answered.get(module).and_then(::serde_json::Value::as_str),
                Some("ok"),
                "case {module}: expected the module to import, got {answered}"
            );
        }
        // The type `os` re-exports from the module it is built on, read by name, by index
        // and by length.
        let built = "import os\n\
                     def handler(event, context):\n    \
                         read = os.stat_result((33188, 2, 3, 1, 0, 0, 12, 1.0, 2.0, 3.0))\n    \
                         return [read.st_mode, read.st_size, read[6], len(read)]\n";
        let finished = ran(built, "lambda_function.handler", None);
        let answered = returned(&finished, "the stat_result type");
        assert_eq!(
            answered,
            ::serde_json::json!([33188, 12, 12, 10]),
            "case os.stat_result: expected the stat fields, got {answered}"
        );
    }

    /// A standard-library module that WORKS, not only one that imports: the archive writer
    /// builds a package in memory and the archive reader reads it back, which is what a
    /// handler doing ordinary file work does.
    ///
    /// The reader decodes an entry's name with the codec the ZIP format names for a name not
    /// flagged UTF-8, so this row also pins that a text codec is reachable BY NAME from code
    /// that imported nothing of the codec machinery itself
    /// ([`super::install_codec_registry`]).
    #[test]
    fn the_standard_librarys_archive_machinery_round_trips_inside_a_run() {
        let source = "import io\n\
                      import zipfile\n\
                      def handler(event, context):\n    \
                          buffer = io.BytesIO()\n    \
                          with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as built:\n        \
                              built.writestr('rows.txt', 'id,total\\n1,42\\n')\n    \
                          package = buffer.getvalue()\n    \
                          with zipfile.ZipFile(io.BytesIO(package)) as read:\n        \
                              names = sorted(read.namelist())\n        \
                              held = read.read('rows.txt').decode('utf-8')\n    \
                          return {'names': names, 'held': held,\n            \
                                  'single_byte': b'\\x82'.decode('cp437')}\n";
        // Importing the archive machinery (`zipfile`, and `shutil` behind it) is more work
        // than the small budget above allows, and is work a real invocation's own budget
        // covers ([`crate::state::execution`]), so this row is given a run's own.
        let mut whole = budget();
        whole.steps = 20_000_000;
        let finished = ran_with(
            source,
            "lambda_function.handler",
            None,
            &BTreeMap::new(),
            whole,
        );
        let answered = returned(&finished, "the archive round trip");
        assert_eq!(
            answered,
            ::serde_json::json!({
                "names": ["rows.txt"],
                "held": "id,total\n1,42\n",
                "single_byte": "é",
            }),
            "case zipfile: expected the package to be read back as it was built, got \
             {answered}"
        );
    }

    /// The native modules a run can reach are WITHIN the allow-list — read from inside a
    /// run, which is where it matters — and no host-reaching one is among them.
    #[test]
    fn the_registered_native_modules_are_within_the_allow_list() {
        let source = "import sys\n\
                      def handler(event, context):\n    \
                          return sorted(sys.builtin_module_names)\n";
        let finished = ran(source, "lambda_function.handler", None);
        let answered = returned(&finished, "the registered modules");
        let actual: Vec<String> = answered
            .as_array()
            .map(|names| {
                names
                    .iter()
                    .filter_map(::serde_json::Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        for name in &actual {
            assert!(
                ALLOWED_NATIVE_MODULES.contains(&name.as_str())
                    || SUPPLIED_MODULES.contains(&name.as_str()),
                "case {name}: a module outside the allow-list is reachable, got {actual:?}"
            );
        }
        for absent in [
            "_socket",
            "_ssl",
            "select",
            "_posixsubprocess",
            "mmap",
            "_ctypes",
            "_signal",
            "_interpreters",
            "fcntl",
            "resource",
            "pwd",
        ] {
            assert!(
                !actual.iter().any(|name| name == absent),
                "case {absent}: a host-reaching module is registered, got {actual:?}"
            );
        }
    }
}
