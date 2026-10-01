//! Plans the one-call functions keep for reuse.

use std::sync::{Arc, Mutex, PoisonError};

use crate::error::Result;
use crate::plan::{Plan, Spec};

/// Plans kept, most recently used last. GPU plan creation compiles shaders,
/// which a loop of one-call transforms of the same sizes should pay once.
const CAPACITY: usize = 4;

type Entry = (Spec, Arc<Mutex<Plan>>);

static PLANS: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

/// The kept plan for `spec`, or a new one that is then kept.
fn plan_for(spec: Spec) -> Result<Arc<Mutex<Plan>>> {
    {
        let mut plans = PLANS.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(index) = plans.iter().position(|(kept, _)| *kept == spec) {
            let entry = plans.remove(index);
            let plan = Arc::clone(&entry.1);
            plans.push(entry);
            return Ok(plan);
        }
    }
    // Created outside the lock, so other threads' calls are not held up.
    let plan = Arc::new(Mutex::new(Plan::new(spec.clone())?));
    let mut plans = PLANS.lock().unwrap_or_else(PoisonError::into_inner);
    if plans.len() == CAPACITY {
        plans.remove(0);
    }
    plans.push((spec, Arc::clone(&plan)));
    Ok(plan)
}

/// Sets the points and runs one batch on the kept plan for `spec`.
pub(crate) fn run(
    spec: Spec,
    points: &[f64],
    targets: &[f64],
    input: &[f64],
    output: &mut [f64],
) -> Result<()> {
    let plan = plan_for(spec)?;
    let mut plan = plan.lock().unwrap_or_else(PoisonError::into_inner);
    plan.set_points(points, targets)?;
    plan.execute(input, output)
}

pub(crate) fn clear() {
    PLANS.lock().unwrap_or_else(PoisonError::into_inner).clear();
}
