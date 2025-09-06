# Internal Documentation Directory

## ⚠️ IMPORTANT SECURITY NOTICE
This directory contains sensitive implementation details, strategic plans, and internal documentation that should NOT be made public. This entire directory is excluded from git tracking via .gitignore.

## Directory Structure

### `/agents/`
- Agent coordination guides and workflows
- AI agent integration patterns
- Sub-agent usage documentation

### `/architecture/`
- System architecture designs
- Implementation summaries for major features
- Search and ingestion system designs
- Technical deep-dive documentation

### `/deployment/`
- Deployment strategies and infrastructure plans
- Health monitoring configurations
- DevOps and infrastructure documentation

### `/implementation/`
- Feature implementation summaries
- Code generation patterns and schemas
- AI model communication logs
- Development progress reports

### `/planning/`
- Project roadmaps and phase planning
- Development workflows and processes
- Session progress summaries
- Strategic planning documents

### `/testing/`
- Testing strategies and checklists
- Manual testing guides
- Coverage reports and benchmarking
- Quality assurance processes

## Usage Guidelines

1. **Never commit sensitive files** - Always verify files aren't tracked by git
2. **Use internal references** - Reference these docs only in internal communications
3. **Regular cleanup** - Remove outdated planning docs periodically
4. **Access control** - Ensure only authorized team members access these files

## File Naming Convention

For any new internal documentation, use one of these patterns:
- `FILENAME_INTERNAL.md`
- `INTERNAL_FILENAME.md`
- Store directly in appropriate `/internal/` subdirectory

These patterns are automatically ignored by git.