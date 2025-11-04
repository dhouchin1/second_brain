# MCP Server Setup for Second Brain

## Overview

This guide covers setting up **two powerful integrations** for Second Brain:

1. **MCP Server** - Exposes Second Brain tools to Claude Code and other MCP clients
2. **Custom Slash Commands** - Quick commands for common tasks in Claude Code

---

## Part 1: MCP Server Setup

### What is an MCP Server?

An MCP (Model Context Protocol) server exposes your application's functionality to AI assistants like Claude Code. Think of it as an API specifically designed for AI tools.

### Installation

1. **Install MCP package:**

```bash
source .venv/bin/activate
pip install mcp
```

2. **Make server executable:**

```bash
chmod +x mcp_server.py
```

3. **Test the server:**

```bash
python mcp_server.py
```

It should start without errors and wait for input.

### Configuration

Add the MCP server to your Claude Code settings:

**Location**: `.claude/settings.local.json` (or global settings)

```json
{
  "mcpServers": {
    "second-brain": {
      "command": "python",
      "args": [
        "/Users/dhouchin/mvp-setup/second_brain/mcp_server.py"
      ],
      "env": {}
    }
  }
}
```

**Or use the venv python:**

```json
{
  "mcpServers": {
    "second-brain": {
      "command": "/Users/dhouchin/mvp-setup/second_brain/.venv/bin/python",
      "args": [
        "/Users/dhouchin/mvp-setup/second_brain/mcp_server.py"
      ]
    }
  }
}
```

### Restart Claude Code

After adding the configuration:
1. Restart Claude Code completely
2. The MCP server will auto-start when Claude Code launches
3. You should see "second-brain" in the available tools

### Available MCP Tools

Once configured, you'll have these tools available in Claude Code:

#### 1. **search_notes**
Search through your Second Brain notes

```
Search for: "productivity tips"
Limit: 10 results
```

#### 2. **create_note**
Create a new note

```
Content: "Remember to review the quarterly goals"
Title: "Q4 Review Reminder"
Tags: ["work", "planning"]
```

#### 3. **get_note**
Get details of a specific note by ID

```
Note ID: 123
```

#### 4. **get_vault_stats**
Get statistics about your vault

```
(No parameters needed)
```

Returns:
- Total notes count
- Notes by type
- Recent activity
- Top tags

#### 5. **get_tags**
Get all tags and their frequencies

```
Limit: 20
```

#### 6. **get_recent_notes**
Get recently created notes

```
Limit: 10
Days: 7
```

### Using MCP Tools in Claude Code

Example conversation:

```
You: Search my notes for anything about HTMX

Claude: I'll search your Second Brain for HTMX-related notes.
[Uses search_notes tool]
Found 3 results for 'HTMX':
1. **HTMX Implementation Guide** (ID: 456)
   ...

You: Create a note about this conversation

Claude: I'll create a note for you.
[Uses create_note tool]
✅ Note created successfully!
**ID**: 789
**Title**: HTMX conversation summary
...
```

---

## Part 2: Custom Slash Commands

### What are Slash Commands?

Slash commands are quick shortcuts in Claude Code that expand into detailed instructions. They're stored as markdown files in `.claude/commands/`.

### Available Commands

I've created 5 custom slash commands for you:

#### `/search-notes`
Search through Second Brain notes

```
Usage: /search-notes
Then specify what to search for
```

#### `/create-note`
Create a new note with AI enhancements

```
Usage: /create-note
Then provide note content
```

#### `/analyze-vault`
Get comprehensive analytics about your vault

```
Usage: /analyze-vault
Provides stats, trends, and insights
```

#### `/htmx-component`
Create a new HTMX component

```
Usage: /htmx-component
Then specify component details
```

#### `/debug-htmx`
Debug HTMX issues

```
Usage: /debug-htmx
Then describe the issue
```

### Creating Custom Commands

Create a new file in `.claude/commands/`:

```bash
touch .claude/commands/my-command.md
```

**Template:**

```markdown
# My Command Title

Brief description of what this command does.

## Instructions

Detailed instructions for Claude on how to handle this command:

1. Step 1
2. Step 2
3. Step 3

## Code Examples

```python
# Example code
```

## Response Format

How to format the response to the user.
```

### Command Organization

```
.claude/commands/
├── search-notes.md         # Search functionality
├── create-note.md          # Note creation
├── analyze-vault.md        # Analytics
├── htmx-component.md       # Component creation
└── debug-htmx.md          # Debugging help
```

---

## Comparison: MCP vs Slash Commands

| Feature | MCP Server | Slash Commands |
|---------|-----------|----------------|
| **Type** | Live API connection | Static instructions |
| **Data Access** | Real-time database queries | Instructions only |
| **Performance** | Fast, direct access | Requires Claude to run code |
| **Setup** | Requires server config | Just markdown files |
| **Flexibility** | Structured tools | Freeform instructions |
| **Best For** | Searching, creating, stats | Complex workflows, guidance |

**Recommendation:** Use both!
- **MCP** for data operations (search, create, stats)
- **Slash Commands** for guidance and workflows

---

## Testing

### Test MCP Server

```bash
# Start the server manually
python mcp_server.py

# In another terminal, test with a JSON request
echo '{"method": "tools/list"}' | python mcp_server.py
```

### Test Slash Commands

In Claude Code:
1. Type `/search-notes`
2. Press Enter
3. You should see the command expand with instructions

---

## Troubleshooting

### MCP Server Not Appearing

1. **Check configuration path:**
   ```bash
   cat .claude/settings.local.json | grep second-brain
   ```

2. **Test server manually:**
   ```bash
   python mcp_server.py
   # Should start without errors
   ```

3. **Check Claude Code logs:**
   - Look for MCP server startup messages
   - Check for connection errors

4. **Restart Claude Code completely:**
   - Quit and reopen
   - MCP servers start on launch

### Slash Commands Not Working

1. **Check file location:**
   ```bash
   ls -la .claude/commands/
   ```

2. **Verify markdown format:**
   - Must have `.md` extension
   - Must be valid markdown

3. **Restart Claude Code:**
   - Commands are loaded on startup

### Database Access Errors

Make sure the database exists:

```bash
ls -la notes.db
# Should show the database file
```

---

## Advanced Configuration

### Multi-User Support

Modify MCP tools to accept user_id:

```json
{
  "mcpServers": {
    "second-brain": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {
        "DEFAULT_USER_ID": "1"
      }
    }
  }
}
```

### Custom Database Path

Set environment variable:

```json
{
  "mcpServers": {
    "second-brain": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {
        "DB_PATH": "/custom/path/to/notes.db"
      }
    }
  }
}
```

---

## Example Workflows

### Workflow 1: Daily Review

```
1. /analyze-vault
2. View stats and trends
3. [Use MCP: get_recent_notes] to see today's notes
4. Create summary note with /create-note
```

### Workflow 2: Research Session

```
1. /search-notes for topic
2. [Use MCP: search_notes] for related content
3. Review results
4. /create-note with findings
```

### Workflow 3: Component Development

```
1. /htmx-component to get template
2. Implement component
3. /debug-htmx if issues
4. /create-note to document
```

---

## Next Steps

1. ✅ Install MCP package: `pip install mcp`
2. ✅ Add MCP config to `.claude/settings.local.json`
3. ✅ Restart Claude Code
4. ✅ Try `/search-notes` command
5. ✅ Use MCP tools to search your vault
6. 🎯 Create custom commands for your workflows

---

## Resources

- **MCP Documentation**: https://github.com/anthropics/mcp
- **Claude Code Docs**: https://docs.claude.com/claude-code
- **Second Brain API**: Check `app.py` for available endpoints

---

**You're all set!** 🚀

Try saying: "Search my notes for productivity" or use `/analyze-vault` to get started!
