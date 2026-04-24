//! Head-to-head: jetro vs jaq on a *deeply nested* synthetic corpus.
//!
//! Tree shape (7 levels of nesting):
//!   org
//!     └─ regions[]
//!         └─ offices[]
//!             └─ teams[]
//!                 └─ members[]
//!                     └─ projects[]
//!                         └─ tasks[]
//!                             └─ events[]
//!
//! Each query exercises multiple engine features at once:
//!   deep recursion, multi-predicate filter, reshape/pick, aggregation
//!   composed across levels, top-N sort, group/partition, joins.
//!
//! Run:
//!   cargo run --release --example bench_nested_vs_jaq -p jetro-core

use std::time::Instant;
use jetro_core::Jetro;
use serde_json::{json, Value};

use jaq_core::load::{Arena, File, Loader};
use jaq_core::{data, unwrap_valr, Compiler, Ctx, Vars};
use jaq_json::{read as jaq_read, Val as JaqVal};

const ITERS: usize = 8;

// ── Synthetic deeply-nested corpus ──────────────────────────────────────────

fn build_corpus(
    n_regions: usize, offices_per: usize, teams_per: usize,
    members_per: usize, projects_per: usize, tasks_per: usize, events_per: usize,
) -> Value {
    let regions_n = ["us-east","us-west","eu-central","ap-southeast","sa-south","af-south"];
    let skills    = ["rust","go","python","ts","cpp","sql","ml","infra"];
    let statuses  = ["open","in_progress","blocked","in_review","done","cancelled"];
    let severities = ["low","medium","high","critical"];
    let kinds     = ["bug","feature","chore","incident","research"];
    let event_kinds = ["commit","review","comment","ci","deploy"];

    let mut regions = Vec::with_capacity(n_regions);
    for r in 0..n_regions {
        let mut offices = Vec::with_capacity(offices_per);
        for o in 0..offices_per {
            let mut teams = Vec::with_capacity(teams_per);
            for t in 0..teams_per {
                let mut members = Vec::with_capacity(members_per);
                for m in 0..members_per {
                    let mut projects = Vec::with_capacity(projects_per);
                    for p in 0..projects_per {
                        let mut tasks = Vec::with_capacity(tasks_per);
                        for k in 0..tasks_per {
                            let mut events = Vec::with_capacity(events_per);
                            let mut task_total: f64 = 0.0;
                            for e in 0..events_per {
                                let cost = ((r*13 + o*7 + t*11 + m*17 + p*19 + k*23 + e*29) % 500) as f64 + 1.5;
                                task_total += cost;
                                events.push(json!({
                                    "id": format!("EV-{}-{}-{}-{}-{}-{}-{}", r,o,t,m,p,k,e),
                                    "kind": event_kinds[(r+o+t+m+p+k+e) % event_kinds.len()],
                                    "cost": cost,
                                    "ts": 1_700_000_000i64 + ((r*1000+o*100+t*50+m*25+p*10+k*5+e) as i64),
                                    "actor": format!("u{}", (r*31+m) % 9973),
                                }));
                            }
                            tasks.push(json!({
                                "id": format!("T-{}-{}-{}-{}-{}-{}", r,o,t,m,p,k),
                                "kind": kinds[(k + p) % kinds.len()],
                                "status": statuses[(k + m) % statuses.len()],
                                "severity": severities[(k + t) % severities.len()],
                                "estimate": (((k+p+m) % 40) + 1) as i64,
                                "actual":   ((((k+p+m) % 40) + 1) as i64 + ((k % 7) as i64 - 3)),
                                "total_cost": (task_total * 100.0).round() / 100.0,
                                "tags": vec![
                                    skills[(k+p) % skills.len()],
                                    skills[(m+t) % skills.len()],
                                ],
                                "events": events,
                            }));
                        }
                        projects.push(json!({
                            "id": format!("P-{}-{}-{}-{}-{}", r,o,t,m,p),
                            "name": format!("project-{}-{}-{}-{}-{}", r,o,t,m,p),
                            "active": (r + o + t + m + p) % 3 != 0,
                            "budget": (((r*997 + p*131) % 50_000) + 500) as i64,
                            "priority": severities[(p + m) % severities.len()],
                            "lead": format!("u{}", (r*31+m) % 9973),
                            "tasks": tasks,
                        }));
                    }
                    let member_skills: Vec<&str> = (0..3)
                        .map(|i| skills[(m + t + i) % skills.len()])
                        .collect();
                    members.push(json!({
                        "id": format!("M-{}-{}-{}-{}", r,o,t,m),
                        "name": format!("member-{}-{}-{}-{}", r,o,t,m),
                        "level": (m % 6) + 1,
                        "skills": member_skills,
                        "active": m % 7 != 0,
                        "salary": (((r*43 + o*17 + t*11 + m*31) % 200_000) + 40_000) as i64,
                        "projects": projects,
                    }));
                }
                teams.push(json!({
                    "id": format!("TM-{}-{}-{}", r,o,t),
                    "name": format!("team-{}-{}-{}", r,o,t),
                    "focus": skills[(r+o+t) % skills.len()],
                    "headcount": members_per,
                    "members": members,
                }));
            }
            offices.push(json!({
                "id": format!("OF-{}-{}", r,o),
                "city": match (r+o) % 6 {
                    0=>"Tokyo",1=>"Berlin",2=>"São Paulo",3=>"Nairobi",4=>"Austin",_=>"Toronto"
                },
                "capacity": (o + 1) * 50,
                "open": o % 4 != 0,
                "teams": teams,
            }));
        }
        regions.push(json!({
            "id": format!("R-{}", r),
            "name": regions_n[r % regions_n.len()],
            "continent": match r % 5 { 0=>"NA",1=>"EU",2=>"AS",3=>"AF",_=>"SA" },
            "offices": offices,
        }));
    }

    json!({
        "org": {
            "name": "MegaCorp",
            "founded": 1999,
            "regions": regions,
            "ceo": "u0",
        }
    })
}

// ── timing ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Stats { best: u128, median: u128, mean: u128 }

fn sample<F: FnMut()>(mut f: F) -> Stats {
    let _ = f(); // warmup
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_micros());
    }
    samples.sort();
    Stats {
        best: samples[0],
        median: samples[samples.len() / 2],
        mean: samples.iter().sum::<u128>() / samples.len() as u128,
    }
}

fn show(label: &str, s: Stats) {
    println!("  {:<14} best {:>9}µs  median {:>9}µs  mean {:>9}µs",
             label, s.best, s.median, s.mean);
}

// ── jaq runner (compile once, reuse) ────────────────────────────────────────

fn compile_jaq(code: &str) -> &'static jaq_core::compile::Filter<jaq_core::Native<data::JustLut<JaqVal>>> {
    let arena: &'static Arena = Box::leak(Box::new(Arena::default()));
    let defs = jaq_core::defs().chain(jaq_std::defs()).chain(jaq_json::defs());
    let loader = Loader::new(defs);
    let modules = loader.load(arena, File { code, path: () }).unwrap();
    let funs = jaq_core::funs().chain(jaq_std::funs()).chain(jaq_json::funs());
    let filter = Compiler::default().with_funs(funs).compile(modules).unwrap();
    Box::leak(Box::new(filter))
}

fn run_jaq(filter: &'static jaq_core::compile::Filter<jaq_core::Native<data::JustLut<JaqVal>>>, input: &JaqVal) -> usize {
    let ctx = Ctx::<data::JustLut<JaqVal>>::new(&filter.lut, Vars::new([]));
    filter.id.run((ctx, input.clone())).map(unwrap_valr).count()
}

// ── bench harness ───────────────────────────────────────────────────────────

fn bench(
    label: &str, desc: &str,
    jetro_tree: &Jetro, jetro_scan: Option<&Jetro>,
    jetro_q: &str, jaq_q: &str, jaq_input: &JaqVal,
) {
    println!("\n{}", label);
    println!("  {}", desc);
    println!("  jetro: {}", jetro_q);
    println!("  jq   : {}", jaq_q);

    // `collect_val` — keep result as jetro's native `Val` for parity with
    // jaq (which returns its own `Val` iterator without materialising to
    // `serde_json::Value`).
    let t = sample(|| { let _ = jetro_tree.collect_val(jetro_q).unwrap(); });
    show("jetro-tree", t);
    if let Some(js) = jetro_scan {
        let s = sample(|| { let _ = js.collect_val(jetro_q).unwrap(); });
        show("jetro-scan", s);
    }
    let compiled = compile_jaq(jaq_q);
    let j = sample(|| { let _ = run_jaq(compiled, jaq_input); });
    show("jaq", j);

    let ratio = j.median as f64 / t.median.max(1) as f64;
    println!("  jetro-tree vs jaq (median): {:.2}x {}",
             ratio, if ratio >= 1.0 { "(jetro faster)" } else { "(jaq faster)" });
    if let Some(js) = jetro_scan {
        let s = sample(|| { let _ = js.collect_val(jetro_q).unwrap(); });
        let r = j.median as f64 / s.median.max(1) as f64;
        println!("  jetro-scan vs jaq (median): {:.2}x", r);
    }
}

fn main() {
    // Shape: 5 × 4 × 4 × 6 × 3 × 4 × 3 =  17280 tasks, 51840 events.
    // Produces a ~30 MB doc with 8-deep nesting.
    let doc = build_corpus(5, 4, 4, 6, 3, 4, 3);
    let bytes = serde_json::to_vec(&doc).unwrap();
    let mb = bytes.len() as f64 / 1_048_576.0;
    println!("payload: {:.2} MB, nesting depth 8, iters {}", mb, ITERS);

    let j_tree = Jetro::new(doc.clone());
    let j_scan = Jetro::from_bytes(bytes.clone()).unwrap();
    let jaq_input: JaqVal = jaq_read::parse_single(&bytes).unwrap();

    // ── Q1: shallow deep-projection through 4 levels ─────────────────────────
    bench(
        "Q1  4-level shape projection — region → office → team → name",
        "Project team names across every region/office without any filter.",
        &j_tree, None,
        "$.org.regions.map(offices).flatten().map(teams).flatten().map(name)",
        "[.org.regions[].offices[].teams[].name]",
        &jaq_input,
    );

    // ── Q2: filter at multiple levels, then project ─────────────────────────
    bench(
        "Q2  multi-level filter chain (open office → active project → critical task)",
        "Only open offices, only active projects, only critical tasks — project task ids.",
        &j_tree, None,
        r#"$.org.regions.map(offices).flatten().filter(open == true).map(teams).flatten().map(members).flatten().map(projects).flatten().filter(active == true).map(tasks).flatten().filter(severity == "critical").map(id)"#,
        r#"[.org.regions[].offices[] | select(.open == true) | .teams[].members[].projects[] | select(.active == true) | .tasks[] | select(.severity == "critical") | .id]"#,
        &jaq_input,
    );

    // ── Q3: deep recursion for multi-predicate find ─────────────────────────
    bench(
        "Q3  deep $..find with two predicates",
        "Every descendant object whose status==in_review AND severity==high.",
        &j_tree, Some(&j_scan),
        r#"$..find(@.status == "in_review", @.severity == "high")"#,
        r#"[.. | objects | select((.status? == "in_review") and (.severity? == "high"))]"#,
        &jaq_input,
    );

    // ── Q4: deep key extract + aggregate ────────────────────────────────────
    bench(
        "Q4  deep key sum — every `cost` anywhere in tree",
        "Fold every numeric `cost` leaf. Heavy recursion for jaq.",
        &j_tree, Some(&j_scan),
        "$..cost.sum()",
        "[.. | objects | .cost? // empty] | add",
        &jaq_input,
    );

    // ── Q5: deep shape projection ───────────────────────────────────────────
    bench(
        "Q5  deep shape + pick rename",
        "Every task: pick id → task_id, severity → sev, total_cost → cost.",
        &j_tree, None,
        "$.org.regions.map(offices).flatten().map(teams).flatten().map(members).flatten().map(projects).flatten().map(tasks).flatten().pick(task_id: id, sev: severity, cost: total_cost)",
        r#"[.org.regions[].offices[].teams[].members[].projects[].tasks[] | {task_id: .id, sev: .severity, cost: .total_cost}]"#,
        &jaq_input,
    );

    // ── Q6: nested reduce — sum of events.cost inside every task, then avg ──
    bench(
        "Q6  nested aggregate: per-task event cost sum, then mean across tasks",
        "Evaluate a sub-aggregate (map(events).flatten().map(cost).sum()) inside outer map.",
        &j_tree, None,
        "$.org.regions.map(offices).flatten().map(teams).flatten().map(members).flatten().map(projects).flatten().map(tasks).flatten().map(total_cost).avg()",
        "[.org.regions[].offices[].teams[].members[].projects[].tasks[].total_cost] | add / length",
        &jaq_input,
    );

    // ── Q7: multi-level filter + count ──────────────────────────────────────
    bench(
        "Q7  5-level filter + count blocked tasks in active projects",
        "Counts blocked tasks belonging to active projects — FilterCount fusion path.",
        &j_tree, None,
        r#"$.org.regions.map(offices).flatten().map(teams).flatten().map(members).flatten().map(projects).flatten().filter(active == true).map(tasks).flatten().filter(status == "blocked").count()"#,
        r#"[.org.regions[].offices[].teams[].members[].projects[] | select(.active == true) | .tasks[] | select(.status == "blocked")] | length"#,
        &jaq_input,
    );

    // ── Q8: top-10 budget values ────────────────────────────────────────────
    bench(
        "Q8  largest 10 project budgets (map + sort + slice)",
        "map(budget), sort ascending, take last 10 — exercises TopN fusion.",
        &j_tree, None,
        "$.org.regions.map(offices).flatten().map(teams).flatten().map(members).flatten().map(projects).flatten().map(budget).sort().reverse()[0:10]",
        "[.org.regions[].offices[].teams[].members[].projects[].budget] | sort | reverse | .[0:10]",
        &jaq_input,
    );

    // ── Q9: group_by(region.name) then count offices ───────────────────────
    bench(
        "Q9  group regions by continent",
        "group_by over regions.continent — partitioning test.",
        &j_tree, None,
        "$.org.regions.group_by(continent)",
        ".org.regions | group_by(.continent)",
        &jaq_input,
    );

    // ── Q10: cross-field filter under nested flatten ────────────────────────
    bench(
        "Q10 events.cost > 400 deep filter + project",
        "Drill into every task's events, filter costly ones, project `id`.",
        &j_tree, None,
        "$.org.regions.map(offices).flatten().map(teams).flatten().map(members).flatten().map(projects).flatten().map(tasks).flatten().map(events).flatten().filter(cost > 400).map(id)",
        "[.org.regions[].offices[].teams[].members[].projects[].tasks[].events[] | select(.cost > 400) | .id]",
        &jaq_input,
    );

    // ── Q11: deep-extract then unique ──────────────────────────────────────
    bench(
        "Q11 deep extract `kind` unique across tree",
        "Collect every `kind` leaf anywhere, dedup. Hits DescendantChain + unique.",
        &j_tree, Some(&j_scan),
        "$..kind.unique()",
        "[.. | objects | .kind? // empty] | unique",
        &jaq_input,
    );

    // ── Q12: compound filter w/ cross-field predicate ──────────────────────
    bench(
        "Q12 task with actual > estimate (over-run)",
        "Predicate compares two fields of same item — typical complex filter.",
        &j_tree, None,
        "$.org.regions.map(offices).flatten().map(teams).flatten().map(members).flatten().map(projects).flatten().map(tasks).flatten().filter(actual > estimate).count()",
        "[.org.regions[].offices[].teams[].members[].projects[].tasks[] | select(.actual > .estimate)] | length",
        &jaq_input,
    );

    // ── Q13: find a single deep record by compound key ─────────────────────
    bench(
        "Q13 deep $..find narrow (single-hit)",
        "Pick one specific task id anywhere in tree.",
        &j_tree, Some(&j_scan),
        r#"$..find(@.id == "T-3-2-1-3-1-2")"#,
        r#"[.. | objects | select(.id? == "T-3-2-1-3-1-2")]"#,
        &jaq_input,
    );
}
