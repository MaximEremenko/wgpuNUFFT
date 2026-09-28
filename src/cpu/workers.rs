//! Scoped worker threads of the CPU plans.
//!
//! Work is split so that every output word is computed by one task in a
//! fixed order, whichever thread runs it: results do not depend on the
//! thread count or on scheduling.

use std::ops::Range;
use std::sync::Mutex;

/// Threads a plan uses unless told otherwise.
pub(crate) fn available_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

/// Splits `len` items into at most `threads` contiguous ranges of at least
/// `min_len` items each (but always one range).
pub(crate) fn ranges(len: usize, threads: usize, min_len: usize) -> Vec<Range<usize>> {
    let groups = threads.min(len / min_len.max(1)).max(1);
    let per_group = len.div_ceil(groups).max(1);
    (0..groups)
        .map(|group| (group * per_group).min(len)..((group + 1) * per_group).min(len))
        .collect()
}

/// Runs `task` for every range, one thread each, and returns the results in
/// range order.
pub(crate) fn map_ranges<R: Send>(
    ranges: &[Range<usize>],
    task: impl Fn(usize, Range<usize>) -> R + Sync,
) -> Vec<R> {
    if ranges.len() <= 1 {
        return ranges
            .iter()
            .enumerate()
            .map(|(group, range)| task(group, range.clone()))
            .collect();
    }
    let task = &task;
    std::thread::scope(|scope| {
        let handles = ranges
            .iter()
            .enumerate()
            .map(|(group, range)| {
                let range = range.clone();
                scope.spawn(move || task(group, range))
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    })
}

/// Runs `task` over contiguous groups of whole `chunk_len` chunks of `data`,
/// on up to `threads` threads, passing the index of each group's first chunk.
/// Every group but the last holds at least `min_chunks` chunks, so small
/// inputs stay on the calling thread.
pub(crate) fn for_each_chunk_group<T: Send>(
    threads: usize,
    data: &mut [T],
    chunk_len: usize,
    min_chunks: usize,
    task: impl Fn(usize, &mut [T]) + Sync,
) {
    let chunks = data.len() / chunk_len;
    let workers = threads.min(chunks / min_chunks.max(1)).max(1);
    if workers == 1 {
        task(0, data);
        return;
    }
    let per_worker = chunks.div_ceil(workers);
    let task = &task;
    std::thread::scope(|scope| {
        for (worker, group) in data.chunks_mut(per_worker * chunk_len).enumerate() {
            scope.spawn(move || task(worker * per_worker, group));
        }
    });
}

/// Runs every task on up to `threads` threads, which take the tasks in order
/// from a shared queue. `init` creates each thread's scratch state.
pub(crate) fn run_tasks<I: Send, S>(
    threads: usize,
    tasks: Vec<I>,
    init: impl Fn() -> S + Sync,
    run: impl Fn(&mut S, I) + Sync,
) {
    let workers = threads.min(tasks.len());
    if workers <= 1 {
        let mut state = init();
        for task in tasks {
            run(&mut state, task);
        }
        return;
    }
    let queue = Mutex::new(tasks.into_iter());
    let (queue, init, run) = (&queue, &init, &run);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(move || {
                let mut state = init();
                loop {
                    let task = queue
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .next();
                    match task {
                        Some(task) => run(&mut state, task),
                        None => break,
                    }
                }
            });
        }
    });
}
