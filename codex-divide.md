# Commercialization Split Plan for Second Brain

## Executive Summary
- **Modular Product Lines**: Seven standalone offerings (Capture, Storage, Search, AI, Experience, Integrations, Ops) outlined in Section 2 with monetization paths aligned to bundles in Sections 5 and 41.
- **Execution Phases**: Five-stage separation roadmap (Section 6) paired with a 6-month solo-friendly track (Section 21.2) and 18-month milestones (Section 19) to pace commercialization.
- **Go-to-Market Core**: Pricing, personas, and messaging pillars consolidated in Sections 41, 46, and 47; discovery scripts and pilot trackers in Sections 42–44 accelerate customer acquisition.
- **Operational Backbone**: Repository strategy (Section 26), backlog rituals (Section 45), metrics dashboards (Section 34), and support processes (Section 30) keep delivery predictable for a small team.
- **Technical Debt Plan**: Cleanup, database modernization, and security standardization tracks captured in Sections 51–58 to ensure the split hardens the codebase rather than fragmenting it.
- **Reference Map**: Use the navigation cues above to jump directly to deeper detail as needs arise; the remainder of this document serves as the full playbook.

## 1. Goals & Guiding Principles
- Create independently salable offerings while preserving a cohesive ecosystem experience.
- Encapsulate bounded contexts (capture, storage, search, AI insight, delivery) to minimize cross-project coupling.
- Standardize data contracts (SQLite schema, event payloads, REST/gRPC APIs) so projects interoperate cleanly.
- Support multiple deployment modes (SaaS, managed on-prem, self-hosted) through consistent packaging and automation.
- Maintain shared developer experience with unified documentation, observability conventions, and release cadence.

## 2. Proposed Product Portfolio

### A. Capture & Ingestion Platform
**Value Proposition**: Turnkey ingestion pipelines for text, web, audio, and advanced media sources.
- Scope: `services/ingestion_queue.py`, `services/archivebox_worker.py`, `processor.py`, `tasks.py`, `tasks_enhanced.py`, `file_processor.py`, integration scripts in `apple_shortcuts/`, `browser-extension/`, `scripts/` for capture utilities.
- Deliverables: REST/gRPC ingestion API, background workers, connector SDK, capture workflow templates, deployment helm charts.
- Interfaces: Emits normalized items onto a message bus or ingestion API contract consumed by downstream projects.
- Monetization: Tiered pricing per connector volume, enterprise connectors catalog, professional services for custom sources.

### B. Knowledge Graph & Storage Engine
**Value Proposition**: Durable knowledge base with rich metadata, graph memory, and lifecycle management.
- Scope: `database.py`, `db/` migrations (including `db/migrations/016_graph_memory.sql`), `graph_memory/`, `note_relationships.py`, data retention utilities.
- Deliverables: Managed SQLite/SQL service with optional Postgres adapter, schema migration tooling, data governance toolkit, backup/restore automation.
- Interfaces: Provides canonical storage APIs (SQL, REST, gRPC) and emits change events for synchronization.
- Monetization: Hosted database plans, compliance add-ons (audit logging, encryption), premium graph analytics features.

### C. Search & Retrieval Service
**Value Proposition**: Hybrid keyword + semantic search with reranking, embeddings, and analytics.
- Scope: `Search System/` modules (`search_engine.py`, `semantic_search.py`, `hybrid_search.py`), `services/search_adapter.py`, `embedding_manager.py`, `sqlite-vec` integration, `test_search_performance.py`.
- Deliverables: Search API microservice, embedding pipeline, index management CLI, relevance analytics dashboard, SDKs (Python/TypeScript).
- Interfaces: Consumes ingested documents via queue or storage API, exposes search endpoints (`api/routes_search.py`), publishes search telemetry.
- Monetization: Usage-based query pricing, SLA-backed enterprise tier, add-on for private model hosting.

### D. AI Insight & Automation Services
**Value Proposition**: LLM-powered summarization, tagging, relationship inference, automation workflows.
- Scope: `llm_utils.py`, `automated_relationships.py`, `auto-seeding` related services, `summarize.py`, integration with Ollama/sentence-transformers, `AI/ML` pipelines.
- Deliverables: Insight generation API, workflow orchestration engine, prebuilt automations, plug-in marketplace for third-party models.
- Interfaces: Listens to storage/search events, writes enriched metadata back via storage API, provides webhook/event interfaces.
- Monetization: Per-insight billing, premium model marketplace, enterprise customization engagements.

### E. Experience Layer (Web, Mobile, Extensions)
**Value Proposition**: User-facing applications delivering capture, browsing, and insight consumption experiences.
- Scope: `templates/`, `static/`, `app.py` (FastAPI web server), `frontend` assets, `browser-extension/`, `Integrations/discord_bot.py`, Apple Shortcuts.
- Deliverables: Web dashboard product, browser extension package, mobile roadmap (Shortcuts, future native apps), SSE real-time channel (`realtime_status.py`).
- Interfaces: Consumes APIs from Capture, Storage, Search, and AI services; offers SSO/Auth integration module.
- Monetization: Seat-based licensing, branded enterprise portals, white-label options for partners.

### F. Integration & Automation Toolkit
**Value Proposition**: Developer-facing tools for integrating Second Brain data with external systems.
- Scope: `api/` modular routers, `config/` templates, `scripts/`, `Integrations/`, `obsidian_sync.py`, `services/obsidian_sync.py`, webhook utilities.
- Deliverables: Unified API gateway, SDKs, event bridge (webhooks, Kafka/Redis adapters), sample automation playbooks, Terraform modules.
- Interfaces: Acts as glue for cross-project orchestration; ensures consistent auth, rate limiting, and governance.
- Monetization: Developer tier subscriptions, marketplace revenue share, professional integration services.

### G. Operations & Observability Suite (Optional but strategic)
**Value Proposition**: Manage deployments, monitoring, compliance for the product suite.
- Scope: `deploy/`, `deployment/`, Dockerfiles, `HEALTH_MONITORING.md`, scripts in `tools/`, instrumentation hooks in services.
- Deliverables: Deployment orchestrator (Terraform/Helm modules), monitoring dashboards, incident response runbooks, licensing enforcement components.
- Monetization: Premium support plans, managed hosting, compliance certifications.

## 3. Repository & Packaging Strategy
- Adopt a polyrepo setup with one primary repo per product line; maintain a lightweight "Second Brain Platform" meta-repo (or manifest) for bundled deployments.
- Extract shared domain models and utilities into a versioned "sb-core" library (Python package + schema files) published to a private registry.
- Standardize API contracts using OpenAPI/gRPC definitions stored in a dedicated "sb-contracts" repo to keep services aligned.
- Provide starter templates (cookiecutter) in each repo to help customers extend functionality without touching core code.
- Introduce cross-project CI/CD pipelines using reusable GitHub Actions workflows stored in a "sb-devops" repository.

## 4. Deployment & Delivery Models
- **SaaS**: Bundle all services behind a unified control plane; orchestrate via Kubernetes with per-tenant namespaces.
- **Managed On-Prem**: Deliver Helm charts/docker-compose bundles for each product; Operations Suite acts as command center.
- **Self-Hosted Community Edition**: Offer trimmed-down bundles (Capture + Storage + Search + minimal AI) via scripted installers.
- **Marketplace/Integrations**: Package browser extension, Shortcuts, Discord bot as standalone downloadable artifacts linked to Integration Toolkit APIs.

## 5. Commercial Bundles & Upsell Paths
1. **Essentials**: Capture Platform + Storage Engine + Search Service (core knowledge base).
2. **Professional**: Essentials + AI Insight Services + Integration Toolkit basic tier.
3. **Enterprise**: Professional + Operations Suite + premium integrations (Discord/Obsidian, custom connectors).
4. **Add-ons**: Advanced AI models, analytics workspace, dedicated support, compliance packages.

## 6. Phased Separation Roadmap
- **Phase 0 (Preparation)**: Inventory shared dependencies, codify contracts, add integration tests per boundary.
- **Phase 1 (Codebase Split)**: Extract repositories sequentially (Capture → Search → AI → Experience → Integrations) while keeping legacy monorepo as reference.
- **Phase 2 (Service Hardening)**: Introduce authentication, rate limiting, billing hooks, and SLA monitoring per service.
- **Phase 3 (Productization)**: Package onboarding flows, documentation portals, licensing checks, and customer analytics dashboards.
- **Phase 4 (Growth)**: Establish partner SDKs, marketplace governance, and joint go-to-market playbooks.

## 7. Risks & Mitigations
- **Coupling Risk**: Tight inter-module imports may slow separation. Mitigation: introduce abstraction interfaces, dependency inversion, and integration tests before extraction.
- **Operational Overhead**: Multiple repos increase release complexity. Mitigation: invest early in automation (CI/CD templates, infrastructure-as-code).
- **Data Consistency**: Cross-service data sync errors could degrade user trust. Mitigation: event sourcing with idempotent consumers, schema versioning, contract tests.
- **Customer Experience Fragmentation**: Multiple products may confuse buyers. Mitigation: curate bundles, maintain unified branding and single sign-on.
- **Resource Constraints**: Splitting requires staffing. Mitigation: prioritize highest-value products (Capture, Search, AI) before secondary offerings.

## 8. Immediate Next Steps
- Map concrete ownership by forming product squads around each proposed project.
- Document API contracts and domain events in "sb-contracts" repo draft.
- Pilot extraction by isolating Search & Retrieval Service as the first standalone repo to validate tooling and deployment approach.
- Create customer-facing value propositions and pricing hypotheses for each bundle to guide engineering investment.

## 9. Cross-Service Interfaces & Contracts
- **Event Bus Vocabulary**: Define canonical events (`capture.item_ingested`, `storage.note_updated`, `search.indexed`, `ai.annotation_created`) with JSON schema versions stored in `sb-contracts`. Include required metadata (tenant, source, timestamps, checksum) to support multi-tenant SaaS.
- **Synchronous APIs**: Publish OpenAPI specs per service; mandate consistent auth (JWT + service-to-service mTLS). Search consumes Storage via `GET /notes/{id}`, AI uses `POST /insights` to write back enrichments.
- **Shared Libraries**: Extract common domain models (Note, Attachment, Relationship, EmbeddingJob) and client SDKs into `sb-core`; release semver tags to coordinate deployments.
- **Data Residency**: Capture + Storage own primary data; downstream services keep derived data caches with explicit TTL and rebuild routines to simplify compliance.
- **Observability**: Standardize tracing headers (`traceparent`, `sb-request-id`) and logging formats so cross-project debugging remains feasible after the split.

## 10. Migration Playbook by Product Line
1. **Capture & Ingestion**
   - Refactor ingestion pipelines to emit to event bus without direct DB writes.
   - Abstract storage writes behind adapter so both monolith and new Storage service can be targeted during transition.
   - Extract connector configuration files and secrets into dedicated config service.
2. **Knowledge Graph & Storage**
   - Freeze schema changes; create migration backlog for required enhancements.
   - Build API facade in monolith that proxies to new Storage service once deployed.
   - Establish database replication/backup strategy before offering hosted plans.
3. **Search & Retrieval**
   - Containerize existing search stack; add ingestion listener that rebuilds FTS + embedding indexes from event bus.
   - Implement contract tests verifying parity with monolithic search endpoints before cutover.
   - Introduce usage metering hooks to support billing.
4. **AI Insight Services**
   - Decouple direct DB reads; rely on Storage and Search APIs for content access.
   - Package model assets separately (Ollama templates, sentence-transformer weights) with licensing review.
   - Add feature flags to toggle legacy vs. service-based enrichment.
5. **Experience Layer**
   - Implement API gateway configuration pointing to modular services.
   - Update auth/session handling to support multi-service backends (central identity provider).
   - Build fallback adapters to call monolith during staged rollout.
6. **Integration Toolkit**
   - Introduce unified developer portal with API keys, docs, and sample apps referencing new endpoints.
   - Provide migration guides for existing Obsidian/Discord integrations to use new contracts.
7. **Operations Suite**
   - Define baseline SLOs per service; integrate monitoring agents before customer rollouts.
   - Automate infrastructure provisioning with Terraform modules referencing separated repos.

## 11. Organizational & Governance Model
- **Product Squads**: Assign cross-functional teams (PM, Lead Engineer, Designer, QA) to each product line with clear KPIs.
- **Platform Team**: Owns `sb-core`, shared tooling, and developer experience; arbitrates breaking changes via architecture review board.
- **Program Management**: Establish commercialization council to synchronize releases, licensing, and GTM campaigns.
- **Support & Success**: Create tiered support structure with shared customer portal spanning all offerings.
- **Revenue Ops Alignment**: Map sales engineers and customer success managers to bundles; define escalation paths for multi-product deals.

## 12. Go-To-Market & Pricing Strategy
- **Target Segments**: Knowledge-heavy SMBs (consulting, legal), innovation teams, and enterprise research groups.
- **Pricing Axes**: Volume (ingestion events, storage size, search queries), seats, premium AI inference minutes.
- **Packaging Tactics**: Offer sandbox environments, usage credits for integrations, and co-marketing with partner ecosystem.
- **Sales Motions**: Self-serve trials for Essentials, assisted sales for Professional, dedicated account teams for Enterprise.
- **Channel Strategy**: Marketplace listings (Slack, Atlassian, Notion), partner resellers for regulated industries, technology alliances with model providers.

## 13. Compliance, Security & Legal
- **Baseline Controls**: SOC2 roadmap, GDPR/CCPA readiness, data encryption in transit and at rest per service.
- **Contractual Assets**: Master Service Agreement templates tailored per bundle; Data Processing Addendum referencing Storage & AI services.
- **Security Architecture**: Central identity provider (OIDC), API gateway enforcing rate limits and WAF rules, secrets management via Vault.
- **Auditability**: Storage maintains immutable audit logs; Operations Suite surfaces compliance dashboards.
- **IP Strategy**: Patent key AI workflows, trademark product names, clarify licensing of open-source dependencies before commercialization.

## 14. Technical Investment Backlog
- **Capture**: Connector SDK v2, retry/guaranteed delivery, ingestion blueprint marketplace.
- **Storage**: Multi-tenant schema support, Postgres-compatible adapter, lifecycle policies (archival tiers).
- **Search**: Query analytics service, personalized reranking, index warm standby for HA.
- **AI**: Model fine-tuning pipeline, human-in-the-loop review UI, cost optimization tooling.
- **Experience**: Role-based access control, mobile-friendly responsive overhaul, in-app marketplace for automations.
- **Integration Toolkit**: CLI scaffolding, Terraform provider, webhook signing verification library.
- **Operations**: Centralized incident response playbooks, SLA dashboard, license enforcement microservice.

## 15. Documentation & Developer Experience
- Launch `docs.secondbrain.dev` with product-specific docs, shared glossary, and changelog aggregator.
- Provide quickstart guides per product with code samples and Postman collections.
- Maintain architecture decision records (ADRs) in each repo; platform team curates cross-project ADR index.
- Offer certification path (Associate, Professional) for partners implementing integrations.
- Integrate doc updates into release checklist; enforce doc linting in CI.

## 16. Metrics & KPIs
- **Capture**: Successful ingestion rate, connector activation velocity, time-to-first-capture for new tenants.
- **Storage**: Data durability incidents, average query latency, backup recovery point & time objectives.
- **Search**: Search success rate, latency percentiles, embedding refresh latency.
- **AI**: Insight adoption rate, human override percentage, cost per insight.
- **Experience**: DAU/MAU, session duration, feature adoption (dashboards, automation triggers).
- **Operations**: Uptime per service, mean time to detect/resolve incidents, compliance audit pass rate.

## 17. Customer Onboarding & Support Flows
- Develop unified onboarding wizard guiding admins through provisioning Capture, Storage, and Search with default automations.
- Provide in-product tour videos and knowledge base articles segmented by persona (Admin, Integrator, Analyst).
- Implement community forum and office hours program; funnel product feedback into quarterly roadmap review.
- Establish SLAs for ticket response/resolve times; integrate support tooling with Ops telemetry for proactive alerts.

## 18. Partner & Ecosystem Expansion
- Curate partner directory with certified implementation partners and technology integrations.
- Publish API usage guidelines and branding assets for co-marketing.
- Launch hackathons/accelerators encouraging extensions built on Integration Toolkit.
- Create revenue-sharing marketplace agreements for third-party connectors and AI models.

## 19. Timeline & Milestones (18-Month Outlook)
- **Quarter 1**: Stand up `sb-core`, finalize contracts repo, pilot search extraction, define pricing models.
- **Quarter 2**: Launch Capture & Search standalone betas, migrate first design partners, complete Ops observability baseline.
- **Quarter 3**: Release AI Insights beta, general availability for Essentials bundle, initiate compliance audits.
- **Quarter 4**: Enterprise packaging with Operations Suite, roll out integration marketplace, start internationalization efforts.
- **Quarter 5**: Expand AI automation catalog, reach 95% automated deployments via Ops suite, finalize SOC2 Type II.
- **Quarter 6**: Evaluate M&A/partnership opportunities, introduce premium analytics workspace, assess IPO/readiness metrics.

## 20. Success Criteria & Exit Conditions
- Each product can be deployed, billed, and supported independently while composing into unified bundles.
- Customer satisfaction scores maintained at or above monolithic baseline; churn <5% annually post-split.
- Engineering velocity sustained (release cadence > once per month per product) with <15% cross-team dependency blockers.
- Revenue targets met per bundle, with at least two net-new revenue streams unlocked (marketplace, managed hosting).
- Organization operates with clear accountability, shared KPIs, and measurable platform reuse across services.

## 21. Lean Execution Strategy for Solo / Small Teams
- Focus on sequencing rather than parallelization; finish one product extraction before starting the next to limit context switching.
- Adopt a "strangler" approach: keep the monolith operational while redirecting individual capabilities to the new services incrementally.
- Optimize for reuse: centralize shared assets in `sb-core` early to avoid duplicate maintenance across repos.
- Automate only critical paths (linting, basic tests) to conserve time; defer complex CI/CD until two services are in production.
- Rely on infrastructure-light deployments (Docker Compose, Fly.io/Render) before investing in Kubernetes or comprehensive IaC stacks.

### 21.1 Minimal Viable Separation Order
1. **Search & Retrieval**: Highest market differentiation and already modular. Extract first to validate tooling and generate early ARR via usage-based pricing.
2. **Capture & Ingestion**: Second extraction enables standalone connector sales and fuels Search service with reliable data flows.
3. **Integration Toolkit (lightweight)**: Provide a thin gateway + SDK wrapper to support early adopters without full platform overhead.
4. **AI Insight Add-on**: Package as optional module that can run alongside Search via shared queues; monetize per insight while keeping ops minimal.
5. **Knowledge Storage (deferred)**: Delay full storage split until customer demand requires multi-tenant hosting or compliance guarantees.
6. **Experience Layer**: Maintain monolithic UI initially; carve out only when multiple backend services stabilize.

### 21.2 6-Month Delivery Targets
- **Month 1**: Stand up `sb-core` package, define event schema draft, containerize Search service, add local dev scripts.
- **Month 2**: Deploy Search microservice (beta) using monolith as data source; create billing prototype (manual invoicing acceptable).
- **Month 3**: Extract ingestion workers into separate repo using lightweight message queue (Redis Streams or simple HTTP callbacks).
- **Month 4**: Launch Integration Toolkit alpha (Python client + webhook examples); document self-host instructions.
- **Month 5**: Wrap AI enrichment jobs as optional service with feature flag; gate advanced models to reduce compute burn.
- **Month 6**: Publish bundled "Essentials" offer (Search + Capture) with combined installer; gather design partner feedback for next phase.

### 21.3 Solo-Friendly Tooling Stack
- **Version Control**: Polyrepo with trunk-based branching; use Git submodules or meta CLI only if necessary.
- **Automation**: Single GitHub Action per repo running lint + unit tests; manual release scripts stored in `scripts/release.sh`.
- **Observability**: Integrate simple logging (structlog) and uptime checks (Healthchecks.io); defer full tracing.
- **Billing & Auth**: Use third-party services (Stripe Checkout, Auth0/Clerk) to avoid building custom systems early.
- **Documentation**: Maintain single mkdocs site with product toggles instead of multiple doc portals.

### 21.4 Resource & Time Budgeting
- Allocate 2 days per week to "platform" work (contracts, shared libs) and 3 days to active product extraction to keep forward momentum.
- Schedule fortnightly release checkpoints to ensure each extracted service can be demoed to prospects or design partners.
- Maintain backlog in lightweight tool (Linear, GitHub Projects) with clear "Now/Next/Later" categories per product line.
- Track personal workload with WIP limits (max 3 concurrent tasks) to prevent burnout.

### 21.5 Fast Feedback Loops
- Recruit 3–5 design partners early; validate Search beta before investing in Storage split.
- Instrument core metrics with minimal tooling (Mixpanel/Amplitude free tier) to confirm adoption.
- Use Loom or short demo videos to keep stakeholders aligned without lengthy documentation cycles.
- Run monthly retrospectives capturing lessons learned, blockers, and adjustments to sequencing.

### 21.6 Exit Criteria for Each Stage
- **Search**: External customer successfully issues queries against standalone service; latency & reliability match monolith baseline.
- **Capture**: At least two connectors operating independently with automated retries and basic monitoring.
- **Integration Toolkit**: Third-party integration built without direct database access; API keys managed through lightweight auth provider.
- **AI**: Minimum viable inference pipeline generating insights for design partner with controllable costs.
- **Storage**: Decision checkpoint—only proceed with full split if >3 customers request dedicated storage or compliance blockers arise.
- **Experience**: UI continues functioning via API gateway; backlog of UI enhancements informs future separation timing.

### 21.7 Risk Management for Small Teams
- **Bus Factor**: Document critical procedures (deploy, rollback, billing) with concise runbooks; store in repo wiki.
- **Time Constraints**: Guard focus time by batching support requests; leverage async communication with customers.
- **Financial Runway**: Prioritize revenue-generating components (Search, Capture) to fund subsequent splits; leverage pre-orders or consulting engagements.
- **Technical Debt**: Keep monolith tests operational to catch regressions while services peel away; schedule quarterly refactors.
- **Vendor Dependence**: Evaluate contingency plans for external services (LLMs, auth providers) and note replacement criteria.

## 22. Lightweight Bundle Strategy
- Offer "Solo Builder Pack" combining Search service docker-compose, Capture worker, and Integration Toolkit CLI for $X/month.
- Provide "Team Starter" bundle with Auth0 integration, Stripe billing, and hosted logging add-ons.
- Use feature flags to gate premium AI/Operations capabilities until staffing allows full support.
- Create upsell path via managed hosting retainer, offering guaranteed response times without building full Ops suite.

## 23. Lean Documentation Checklist
- Single README per repo outlining install, configure, deploy in under 10 minutes.
- Shared "Playbook" doc covering common tasks (deploy update, run migrations, rotate keys).
- Changelog template for fast release notes; distribute via email or Notion page.
- FAQ focusing on deployment, scaling, billing, and roadmap visibility for early customers.

## 24. Minimal Compliance Track
- Start with privacy policy, terms of service, and simple DPA template referencing third-party providers.
- Implement basic access logging and manual export tooling to satisfy data subject requests.
- Use SOC2-lite checklist to assess gaps quarterly; defer formal audit until $ARR threshold is met.
- Maintain incident response checklist; involve customers transparently if issues occur.

## 25. Future Scale Triggers
- **Hiring Trigger**: Add dedicated engineer once two services have paying customers and backlog exceeds sustainable solo velocity.
- **Tooling Upgrade Trigger**: Move to Kubernetes/advanced CI when deployments exceed 2 per week per service.
- **Compliance Trigger**: Pursue SOC2/HIPAA when enterprise pipeline demands it or annual revenue surpasses set goal.
- **Monolithic Sunset Trigger**: Plan full deprecation when 80% of traffic flows through separated services with stable SLAs.


## 26. Suggested Repository Layout & Naming
- `sb-core`: Shared domain models, contracts, utilities; published as Python package `sb_core`.
- `sb-search-service`: Standalone FastAPI service with embedding jobs, queue consumers, CLI for index ops.
- `sb-capture`: Background workers, connector SDK, CLI for ingesting local files; optional `connectors/` submodules.
- `sb-integrations-kit`: API gateway config, client SDKs, example webhooks, developer portal assets.
- `sb-ai-insights` (deferred): LLM pipelines, summarization jobs, feature flag service.
- `sb-experience` (monolith wrapper): FastAPI web UI acting as aggregator; gradually replaced with modular frontend.
- `sb-ops-lite`: Docker Compose templates, deployment scripts, observability snippets.
- Each repo includes `/docs/quickstart.md`, `/scripts/release.sh`, and `/examples/` to encourage consistent onboarding.

## 27. Customer Migration Playbook (Solo-Friendly)
1. **Discovery Call Template**: Determine current usage (capture volume, search queries, integrations) and pain points.
2. **Environment Audit Checklist**: Confirm Python version, database size, optional dependencies, and hardware requirements.
3. **Pilot Plan**:
   - Week 1: Enable Search service beta with read-only mirror of monolith DB.
   - Week 2: Route subset of traffic (internal users) through new endpoints; collect latency metrics.
   - Week 3: Transition ingestion for one connector to new Capture service; monitor event queue health.
4. **Rollback Procedures**: Document quick toggle to revert API gateway to monolith endpoints; maintain backup snapshots.
5. **Communication Cadence**: Weekly updates, Slack/Discord channel for feedback, shared dashboard of rollout metrics.
6. **Exit Survey**: Capture qualitative feedback, feature requests, and willingness to upgrade to paid bundle.

## 28. Financial & Funding Milestones
- **Baseline Costs**: Track monthly spend for hosting, third-party APIs (LLM, auth, billing), and tooling subscriptions.
- **Revenue Goals**:
  - Month 2: $1k MRR via Search beta design partners.
  - Month 4: $3k MRR with Capture + Search bundle.
  - Month 6: $5k MRR including AI add-on upsells.
- **Cash Flow Tips**: Offer annual prepay discounts, bundle consulting packages, leverage open-source community for marketing.
- **Funding Strategy**: Remain bootstrapped until MRR ≥ $8k and healthy pipeline; prepare investor narrative focusing on modular AI knowledge platform.
- **Pricing Experiments**: Use tiered usage-based pricing with transparent overage rates; test with 3 pilot customers before broad launch.

## 29. Marketing & Launch Assets
- **Positioning Statements**: One-liners per product (Search: "Hybrid semantic search in minutes").
- **Landing Pages**: Simple static site per product using shared template; collect waitlist emails.
- **Demo Materials**: Two-minute overview video, product walkthrough deck, case study template.
- **Content Calendar**: Bi-weekly blog posts showcasing use cases; cross-post on LinkedIn/Dev.to.
- **Community Engagement**: Host monthly live session, share roadmap in public Notion, solicit feature votes.
- **KPIs**: Website conversion rate, waitlist growth, demo-to-close ratio, churn of early adopters.

## 30. Support & Maintenance Operations
- **Support Channels**: Dedicated email alias + Discord server; document SLA (e.g., 24h response for paid plans).
- **Incident Workflow**: Define severity levels, status page template, postmortem checklist.
- **Knowledge Base**: Start with FAQ, common troubleshooting scripts, and video snippets; update quarterly.
- **Maintenance Windows**: Reserve weekly slot for upgrades and refactors; notify customers ahead of time.
- **Backup Owner**: Identify trusted contractor or collaborator who can assist during vacations or overload periods.
- **Tooling**: Use HelpScout/Zendesk lite or shared inbox with tagging; integrate with issue tracker for visibility.


## 31. Experiment Backlog & Validation Framework
- Maintain a living "Growth Experiments" board with columns: Hypothesis, Setup, Result, Decision.
- Use the ICE scoring model (Impact, Confidence, Effort) to rank experiments; limit to 3 active tests at once.
- Core experiment themes:
  - Pricing sensitivity (trial length, usage tiers, concierge onboarding).
  - Feature adoption (search relevancy tweaks, connector catalog expansion).
  - Messaging resonance (landing page headlines, case studies, product naming).
  - Channel efficiency (founder-led sales vs. inbound content vs. community partnerships).
- Success measurement: pick a single north-star metric per experiment (conversion %, activation time, net revenue) to avoid analysis paralysis.
- Post-experiment retro template: What happened, what surprised you, what will you change next.

## 32. Contractor & Partner Engagement Plan
- Identify specialist gaps (design, DevRel, compliance) and prepare short engagement briefs with clear deliverables and budgets.
- Build bench of contractors via trusted networks; keep NDAs and short-form agreements ready.
- Pilot engagements with milestone-based payments to protect cash flow.
- Consider rev-share or advisor equity for high-leverage partners (e.g., industry domain experts) to extend reach without full-time hires.
- Document partner onboarding kit: product overview, messaging guide, integration checklist.

## 33. Customer Success Flywheel
- **Onboarding**: 30-minute kickoff, shared success plan (goals, milestones, ROI metrics), Slack/Discord invite.
- **Activation**: Weekly check-ins during first month, quick wins prioritized (search dashboard, first automation).
- **Adoption**: Monthly usage review, automated health score combining ingestion volume, search frequency, and insight usage.
- **Advocacy**: Invite high-health customers to contribute testimonials, beta programs, and referral incentives.
- Maintain customer journey map in Notion; update playbook quarterly based on feedback.

## 34. Operational Dashboard Blueprint
- Minimum metrics per product surfaced via simple dashboards (Google Data Studio/Airtable):
  - Capture: events ingested/day, failure rate, queue depth.
  - Search: query latency, result click-through, embedding backlog.
  - AI: inference cost/day, insight acceptance rate.
  - Support: tickets opened/closed, SLA adherence, NPS survey.
- Automate data export scripts weekly; manual spreadsheet acceptable initially but plan automation backlog.
- Define alert thresholds (e.g., queue depth > 1000) triggering notifications via Slack/Email.

## 35. Strategic Options & Pivot Scenarios
- **Vertical Solution**: Package services for specific industries (legal research vault) if horizontal growth stalls.
- **API Platform Focus**: Emphasize Integration Toolkit + Search as backend infrastructure if UI adoption lags.
- **Insight Automation Studio**: Expand AI service into workflow builder if automations outperform search revenue.
- **Acquisition Readiness**: Keep clean financials, documentation, and key metrics to stay attractive for strategic buyers.
- Conduct biannual strategy review assessing traction vs. pivots; document decisions in leadership journal.

## 36. Team Growth Readiness Checklist
- Role scorecards ready for first hires (Backend Engineer, Customer Success, Marketing).
- Interview question bank focused on autonomy, cross-domain skills, and startup resilience.
- Compensation bands benchmarked against remote-first startups; include equity guidelines.
- Internal onboarding doc covering architecture, coding standards, deployment rituals, and cultural values.
- Define lightweight performance review cadence (quarterly goal check-ins, feedback loops).

## 37. Founder's Routine & Sustainability Tips
- Implement weekly CEO day for strategy, finances, and partnerships separate from execution tasks.
- Protect deep-work blocks (e.g., mornings) for engineering; schedule meetings/partner calls in afternoons.
- Leverage async status updates (Loom, Notion) to reduce meeting load while keeping stakeholders engaged.
- Track personal KPIs: energy, focus, and satisfaction; adjust workload or delegate when indicators dip.
- Pre-plan rest periods aligned with release cycles to avoid burnout (e.g., 1 week off after major launch).

## 38. Legal & Administrative Checklist
- Incorporate or review existing entity status; ensure IP assignment agreements signed.
- Register trademarks for product names; check domain availability for each product line.
- Maintain master subscription agreement template with modular exhibits per product.
- Confirm export control compliance for AI models; document usage policies.
- Set reminder cadence for annual filings, tax payments, and insurance renewals.

## 39. Public Roadmap & Transparency Practices
- Host public roadmap (Notion/Trello) segmented by `Now`, `Next`, and `Later` to align expectations with the community.
- Share monthly changelog blog posts summarizing releases, experiments, and lessons learned.
- Provide a voting/comment mechanism for feature requests and feed outcomes into prioritization sessions.
- Publish quarterly "State of Second Brain" updates covering product progress, key metrics, and upcoming themes.

## 40. Exit Readiness Signals
- Criteria to consider wind-down or acqui-hire: sustained negative growth, inability to hit revenue milestones, or loss of key partners.
- Maintain clean codebase, documentation, and customer contracts to reduce due diligence friction for potential buyers.
- Keep inventory of intangible assets (brand assets, testimonials, integrations) for valuation discussions.
- Establish an ethical sunset plan ensuring customer data portability and honoring support commitments.

## 41. Pricing & Packaging Matrix (Draft)
| Bundle | Target Persona | Core Inclusions | Usage Limits | Intro Price (USD) | Notes |
| --- | --- | --- | --- | --- | --- |
| Solo Builder | Indie researcher / consultant | Search service container, Capture worker, Integration CLI | 100k documents, 50k queries/mo | 199/mo | Add-on: concierge onboarding $499 one-time |
| Team Starter | 3-10 person innovation team | Solo Builder + Auth0 integration, shared dashboards, support SLA 24h | 250k documents, 150k queries/mo | 499/mo | Volume overages billed at $0.75 per 1k queries |
| Pro Insights | Knowledge ops lead | Team Starter + AI insights service, workflow templates | 400k documents, 250k queries/mo, 10k insights/mo | 899/mo | Additional insights $0.12 each |
| Enterprise | Regulated industries | All services + Ops Lite + custom connectors | Custom | Custom | Includes dedicated success manager |

- Validate willingness to pay via design partner interviews; adjust limits/pricing after first three contracts.
- Offer quarterly/annual prepay discounts (10%/20%) once billing flow stabilized.

## 42. Discovery Call Script Outline
1. **Context & Goals**
   - "Walk me through your current knowledge capture workflow."
   - "What prompted you to look at Second Brain now?"
2. **Pain & Urgency**
   - "Where do things break today—capture, search, or insights?"
   - "How do these issues impact your team (time, errors, missed opportunities)?"
3. **Volume & Constraints**
   - "How many new items are captured weekly? What formats?"
   - "Any compliance or data residency requirements?"
4. **Success Criteria**
   - "What would a successful rollout look like in 90 days?"
   - "Who needs to see value first?"
5. **Timeline & Budget**
   - "When do you need a solution in place?"
   - "Have you scoped budget for this initiative?"
6. **Next Steps**
   - Recap proposed bundle, pilot plan, and decision timeline.
   - Schedule follow-up demo or technical deep dive.

## 43. Design Partner Pilot Tracker (Template)
| Partner | Bundle | Start Date | Current Phase | Key Metrics | Blockers | Next Check-in | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Acme Legal | Solo Builder | 2025-02-10 | Search Beta | 12k queries/day, p95 420ms | Need PDF OCR accuracy review | 2025-02-17 | Add ingestion script training |
| Innovate Labs | Team Starter | 2025-03-01 | Capture Migration | 5 connectors live, 1% failure | Awaiting SSO config | 2025-03-05 | Provide Auth0 config walkthrough |

- Maintain tracker in Notion/Sheets; review weekly during CEO day.
- Capture qualitative feedback column after each check-in to inform roadmap.

## 44. Week-by-Week Execution Checklist (First Quarter)
- **Week 1**: Finalize `sb-core` repo structure, publish v0.1 package, draft pricing page skeleton.
- **Week 2**: Containerize Search service, set up simple billing ledger spreadsheet, conduct 2 discovery calls.
- **Week 3**: Deploy Search beta environment (Fly.io/Render), run latency tests, recruit first design partner.
- **Week 4**: Draft developer docs for Search API, release Loom demo, send follow-up emails to prospects.
- **Week 5**: Extract Capture worker, connect to Redis queue, instrument basic logging.
- **Week 6**: Pilot Capture with internal data, document runbooks, update pricing assumptions.
- **Week 7**: Launch Integration Toolkit alpha, publish webhook example, onboard second design partner.
- **Week 8**: Conduct retrospective, refine experiment backlog, prep AI service roadmap.
- **Week 9**: Wrap AI feature flag scaffolding, run cost analysis for inference providers.
- **Week 10**: Roll out combined Essentials installer to design partners, collect NPS pulse.
- **Week 11**: Focus on support processes—set up ticketing tool, craft FAQ articles.
- **Week 12**: Review metrics, adjust revenue targets, plan next quarter backlog.

## 45. Backlog Prioritization Board Structure
- Columns: `Now`, `Next`, `Later`, `Blocked`, `Done`.
- Card template includes: Objective, Impact score, Effort estimate, Dependencies, Customer signal, Status owner.
- Weekly ritual: move no more than 5 items into `Now`; archive completed cards with short retrospective note.
- Use labels for product lines (Search, Capture, AI, Integrations, Experience, Platform) to balance workload.

## 46. Customer Persona Snapshots
- **Research Ops Lead (Alex)**: Needs reliable capture/search for cross-functional teams; values fast deployment and compliance assurances. Priority bundle: Team Starter.
- **Founder/Consultant (Riley)**: Solo user managing large knowledge base; prioritizes affordability and simplicity. Priority bundle: Solo Builder.
- **Knowledge Manager in Enterprise (Jordan)**: Oversees knowledge strategy for 200+ staff; requires integrations, AI insights, managed support. Priority bundle: Enterprise.
- **Developer Advocate (Casey)**: Builds integrations on behalf of partners; needs API clarity and sandbox environments. Priority focus: Integration Toolkit.

## 47. Messaging Pillars & Proof Points
- **Unified Knowledge Capture**: "Ingest from web, audio, and documents with zero infrastructure". Proof: Show archivebox worker + ingestion queue metrics.
- **Search that Understands Context**: "Hybrid semantic + keyword search tuned for recall and precision". Proof: Benchmark against industry queries.
- **Actionable AI Insights**: "Summaries, tags, and relationships that accelerate decision-making". Proof: Case study highlighting AI-created clusters.
- **Developer-Ready Platform**: "APIs, SDKs, and automation hooks that integrate with your stack". Proof: Reference integration toolkit samples.

## 48. Key Collateral To Produce
- Product one-pagers (PDF) per bundle.
- Technical architecture overview slide (monolith to modular evolution).
- ROI calculator spreadsheet (time saved, cost avoided).
- Security whitepaper summarizing data handling, encryption, and compliance posture.
- Case study template highlighting problem, solution, results, quotes.

## 49. Community & Content Calendar (Quarterly)
- **Month 1**: Launch blog on knowledge ops trends (2 posts), host kickoff webinar, share Search beta announcement.
- **Month 2**: Publish customer interview podcast episode, release connector how-to guide, run AMA in Discord.
- **Month 3**: Release quarterly "State of Second Brain" update, publish technical deep dive on event contracts, announce upcoming AI features.
- Track engagement metrics (registrations, downloads, community growth) to guide future topics.

## 50. Tooling & Resource Wishlist
- Upgrade to paid logging/monitoring tool once revenue covers cost (e.g., Better Stack).
- Evaluate lightweight CRM (Close/Streak) for managing leads and pipeline.
- Acquire design template pack or hire contractor for branding consistency.
- Budget for targeted ads or sponsorships once attribution model validated.

## 51. Codebase Cleanup Roadmap (Supernova Alignment)
- **Phase 0 – Quick Wins (Week 1-2)**: Remove backup/duplicate files (`config_backup.py`, `file_processor_backup.py`, `.backup` artifacts) and document deletions to reduce confusion.
- **Phase 1 – Structure Cleanup (Week 3-4)**: Archive or delete unused directories (`archive/services_unused/`), consolidate standalone tests into `tests/`, and ensure pytest picks up only relevant modules.
- **Phase 2 – Service Inventory (Week 5-6)**: Audit orphaned services (e.g., `services/demo_data_router.py`, `services/raindrop_client.py`) and decide to integrate, document, or remove.
- **Phase 3 – Org-Wide Standards (Week 7-10)**: Standardize import patterns, clean commented-out code, and align configuration loading to single source (`config.py`).
- Track removals/testing in checklist tied to backlog board with "Cleanup" label to maintain visibility.

## 52. Database Connection Modernization Plan
- Inventory all `sqlite3.connect()` usages (172+ instances) and categorize by module (Capture, Search, AI, Integrations).
- Introduce shared connection utility in `sb-core` with context manager and pooling support; provide migration guide for each service.
- Refactor high-traffic paths first (search indexing, capture ingestion) to use the shared utility; schedule 1–2 modules per sprint.
- Add unit tests ensuring connections close properly and smoke tests monitoring concurrency behavior.
- Once refactor reaches 80% coverage, deprecate legacy helpers and add lint rule to flag new raw connections.

## 53. Security & Rate Limiting Standardization
- Adopt a single rate-limiting middleware (e.g., SlowAPI) configured via `sb-core` util; document default policies per router.
- Centralize security headers and CSRF validation in reusable FastAPI dependency; retrofit existing endpoints as they move into separate services.
- Define authentication patterns (JWT + service keys) and ensure Integrations Toolkit references shared implementation.
- Add security checklist to release process verifying headers, rate limits, and auth configs before deploying new services.

## 54. Testing & Utility Script Governance
- Move ad-hoc root-level test scripts into `tests/manual/` or retire them; ensure pytest discoverability via `pytest.ini`.
- Create `scripts/README.md` cataloguing utility scripts, usage status, and owners; remove obsolete debugging tools after verification.
- Introduce pre-commit hook warning when adding root-level test files to maintain structure.
- Align design partner pilot tests with standardized fixtures to reduce duplication between monolith and service repos.

## 55. Architecture Consolidation Tasks
- **Search Consolidation**: Map differences between legacy `search_engine.py` and unified adapters; plan deprecation timeline aligned with Search service extraction.
- **Service Registration**: Document pattern for enabling/disabling routers in monolith vs. new services; remove commented imports once parity achieved.
- **Configuration Harmonization**: Ensure all services load from shared config schema; add validation tests.
- **Documentation Update**: Append cleanup outcomes to developer docs so future contributors understand the new structure.

## 56. Cleanup Backlog Task Ideas
- **Task Cards to Create**:
  - Remove redundant backups (`config_backup.py`, `file_processor_backup.py`) – Owner: Platform lead – Est: 0.5 day – Verify by running regression suite.
  - Consolidate root-level tests into `tests/` with updated `pytest.ini` – Owner: QA – Est: 1 day – Acceptance: pytest collects only intended suites.
  - Retire `archive/services_unused/` by either documenting reference value or deleting – Owner: Capture engineer – Est: 1 day – Include changelog note.
  - Audit utility scripts; produce `scripts/README.md` and mark deprecated tools – Owner: Ops – Est: 1 day – Acceptance: README lists status per script.
  - Draft standard import/style guide and lint rule PR – Owner: Platform – Est: 1.5 days – Acceptance: pre-commit enforces rule set.
  - Clean commented router imports in `app.py` once equivalent service endpoints exist – Owner: Web – Est: 0.5 day – Acceptance: app passes lint/tests.
- Schedule these tasks across Weeks 1-4 with no more than two cleanup cards active simultaneously to protect feature velocity.
- Track progress in backlog board under "Cleanup" label with linked test evidence.

## 57. Database Connection Audit Playbook
- **Preparation**: Run `rg "sqlite3.connect" -n` and export results to spreadsheet with columns (File, Module, Service Area, Notes).
- **Categorize**: Tag each occurrence as Read-only, Write-heavy, Long-running task, or Legacy. Prioritize write-heavy and multi-threaded contexts.
- **Sampling**: For top 10 critical paths (e.g., `search_engine.py`, `processor.py`), document current connection lifecycle and error handling.
- **Migration Plan**:
  - Draft `sb-core` connection helper interface (context manager + pooling stub) and share with team for feedback.
  - Create pilot refactor branch focusing on Search indexing pipeline; measure latency/memory before and after.
  - Produce "refactor checklist" template (update imports, replace context, add tests) reused per module.
- **Validation**: Add smoke test that spins up multiple concurrent ingestion jobs to monitor connection contention; log metrics for before/after comparison.
- **Documentation**: Publish confluence/Notion page summarizing findings, prioritization matrix, and refactor schedule to keep stakeholders aligned.

## 58. Rate Limiting & Auth Stack Options
- **Option A – SlowAPI + Auth0**:
  - Pros: Familiar to FastAPI, quick setup, integrates with Auth0 rules; good for SaaS multi-tenant.
  - Cons: Python-side enforcement only; may require scaling adjustments under heavy load.
- **Option B – API Gateway (Kong/Traefik) + JWT service keys**:
  - Pros: Centralized policy management, supports burst control, plugs into future polyrepo services.
  - Cons: Additional infrastructure to manage; steeper learning curve for solo operator.
- **Option C – Cloudflare Turnstile/Workers edge rate limiting**:
  - Pros: Offloads traffic filtering, built-in bot mitigation; minimal infrastructure overhead.
  - Cons: Depends on external network service; may complicate on-prem deployments.
- **Recommended Path**: Start with Option A for speed, design hooks compatible with Option B migration when services scale.
- **Action Items**:
  - Define default policies (e.g., 60 requests/minute per API key) and document override process.
  - Implement shared dependency injecting auth/rate-limit context into services; create integration tests verifying 429 responses.
  - Update onboarding docs guiding customers through API key generation and auth flow.
