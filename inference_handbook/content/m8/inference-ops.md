<div class="chapter-meta">Module 8 · Chapter 7</div>

# InferenceOps and Production Lifecycle Management

<p class="lede">InferenceOps is MLOps for serving: the end-to-end lifecycle of an LLM in production. Models get released, deployed, canaried, rolled back, versioned, retired. Engineers get paged, incidents happen, post-mortems get written. This chapter is the operational maturity ladder.</p>

## The lifecycle in one picture

```
  Model registry ← training team
        │
        ▼
     Staging      ← engine config, eval pass
        │
        ▼
    Canary (5%)   ← traffic splits, quality monitor
        │
        ▼
  Rollout (100%)  ← production SLO
        │
        ▼
  Retire previous ← after stability period
```

Each step has a gate. Failing a gate means rolling back one step.

## Model versioning

Every deployed model has a version identifier:

```
llama-3.1-70b-instruct-fp8-v2.4.1
```

Breakdown:

- Base model: `llama-3.1-70b-instruct`
- Quantization: `fp8`
- Version: `v2.4.1` (major.minor.patch)

Major = architecture change. Minor = tuning / recipe change. Patch = bugfix (rare for weights).

Store all three fields in your request logs. When a question comes up ("why did this request produce X?"), you need to know the exact model + version that served it.

## Rollout patterns

### Rolling update (default)

K8s updates pods one at a time. During rollout, both old and new are serving. Fastest; some risk.

### Blue-green

Full new fleet stands up in parallel; traffic cuts over at the gateway. Zero downtime; 2× resource cost during cutover. Rollback = flip the gateway back.

### Canary

Deploy to 5–10 % of replicas first. Observe for N hours. If metrics are clean, roll to 100 %. If not, roll back the canary. The standard pattern for frontier model updates.

### Shadow

Send a copy of production traffic to the new model; discard its output. Compare responses offline. No user impact; catches issues before canary.

## Health checks and readiness probes

Every pod should have:

- **Liveness** — process is alive. Hit `/health`. If it fails, K8s restarts the pod.
- **Readiness** — pod is accepting traffic. More thorough — model loaded, first inference succeeded, warm-up complete.

Example readiness probe:

```yaml
readinessProbe:
  httpGet: {path: /health, port: 8000}
  initialDelaySeconds: 60     # model loads slowly
  periodSeconds: 10
  failureThreshold: 3
```

Readiness failing briefly during degradation is expected; the LB stops routing until recovered.

## Canary metrics — what to watch

For the first hour of a new canary:

- **P99 TTFT** — if 20 % worse than baseline, roll back.
- **P99 TPOT** — same threshold.
- **Error rate** — any 5XX increase is a rollback signal.
- **Downstream quality** — if you have real-time quality signals (user thumbs-up, judge scores), watch for regression.
- **Token usage** — a verbose new model can double your bill.

Automate rollback where possible. Humans at 3 AM make slower decisions.

## Incident response

When something breaks in production:

1. **Acknowledge** — someone owns the incident.
2. **Mitigate** — stop the bleeding. Roll back, scale up, shed traffic.
3. **Investigate** — logs, traces, metrics. Not during mitigation.
4. **Communicate** — status page, Slack, stakeholders.
5. **Resolve** — permanent fix shipped.
6. **Post-mortem** — what happened, why, what prevents recurrence.

For LLM-specific incidents, common classes:

| Incident | First action |
|---|---|
| High TTFT spike | Check autoscaler; scale up manually |
| High error rate | Check pod logs; roll back last deploy |
| OOM on replicas | Reduce max_num_seqs or restart with less concurrency |
| Quality regression on new model | Roll back; investigate offline |
| Cost spike | Enable cheaper model routing; investigate traffic source |

## SLO engineering

Don't invent SLOs in isolation. Three questions:

1. **What do users feel?** TTFT &lt; 1 s for chat; &lt; 5 s for long analyses.
2. **What can we actually deliver at reasonable cost?** Measure, don't guess.
3. **What's the gap?** That's your optimization roadmap.

Publish your SLOs internally. If you can't meet them, that's a capacity, cost, or architecture conversation — not a hidden engineering debt.

## A maturity ladder

| Level | What you have |
|---|---|
| 0 | One vLLM pod, no monitoring, deploy by SSH |
| 1 | K8s Deployment, basic Prometheus, alerts on uptime |
| 2 | Canary rollouts, autoscaling, dashboards per service |
| 3 | SLOs with alerts, automated rollback, cost attribution |
| 4 | Multi-region, multi-tenant, A/B testing framework, LLM-as-judge quality monitoring |
| 5 | Everything at 4 + capacity planning automated, model lifecycle policies |

Most teams live at 2–3. The jump to 4–5 is specialized but valuable at scale.

## Pitfalls

- **Rolling out on Fridays.** The classic. Don't.
- **No rollback plan for weight updates.** You need the previous checkpoint ready to serve on short notice.
- **Alerts that fire but don't say what to do.** Every alert should link to a runbook.
- **No post-mortem culture.** Incidents repeat if the lessons don't stick.
- **Treating model updates as deploys.** They're more like database migrations — backward-compatible until verified.

<section class="faq">
<h2>FAQ</h2>
<h3>How long should a canary run?</h3>
<p>At least one full traffic cycle (24 h for consumer apps). Longer for high-risk changes (new model architecture).</p>
<h3>Can I use GitOps for model deployments?</h3>
<p>Yes — ArgoCD / Flux with a model manifest repo. Model version is just another artifact.</p>
<h3>Is there an LLM-specific incident management tool?</h3>
<p>Not really. Standard SRE tools (PagerDuty, Opsgenie, Incident.io) work. The LLM-specific bits are the runbooks.</p>
<h3>How do I retire a model?</h3>
<p>Announce it, deprecate it, route traffic away over 2–4 weeks, then shut down. Give customers time.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://sre.google/sre-book/table-of-contents/" target="_blank" rel="noreferrer">Google SRE Book — foundational material</a></li>
<li><a href="https://argo-cd.readthedocs.io/" target="_blank" rel="noreferrer">ArgoCD — GitOps continuous delivery</a></li>
<li><a href="https://openai.com/policies/sharing-publication-policy" target="_blank" rel="noreferrer">OpenAI — Example SLO disclosures</a></li>
</ul>
</section>

## What's next

Module 8 is done. [Module 9 — Cost & Energy-Efficient Inference](?module=m9&chapter=inference-economics) wraps the handbook with the economics side: dollars, joules, and the capstone design chapter.
