# Custom Claude Code Commands for Second Brain

## Available Commands

Type these commands in Claude Code to get instant help:

### 📝 Note Management

- **`/search-notes`** - Search through your Second Brain notes
  - Uses FTS5 for fast full-text search
  - Returns ranked results with previews

- **`/create-note`** - Create a new note with AI enhancements
  - Auto-generates title if not provided
  - Extracts tags automatically
  - Creates AI summary

### 📊 Analytics

- **`/analyze-vault`** - Get comprehensive vault analytics
  - Note counts and types
  - Tag frequency analysis
  - Activity trends
  - Knowledge graph insights

### 🛠️ Development

- **`/htmx-component`** - Create a new HTMX component
  - Template generation
  - API endpoint code
  - Integration examples
  - TailwindCSS styling

- **`/debug-htmx`** - Debug HTMX issues
  - Systematic troubleshooting
  - Common issue checklist
  - Browser console helpers
  - Network request debugging

## How to Use

1. **Type the command** in Claude Code (e.g., `/search-notes`)
2. **Press Enter** - Command expands with instructions
3. **Provide details** - Claude will ask for any needed information
4. **Get results** - Claude executes the command

## Examples

### Search Notes
```
You: /search-notes
Claude: What would you like to search for?
You: productivity tips
Claude: [Searches and displays results]
```

### Create Note
```
You: /create-note
Claude: What content should I add to the note?
You: Remember to review quarterly goals next week
Claude: [Creates note with AI-generated title and tags]
```

### Analyze Vault
```
You: /analyze-vault
Claude: [Shows comprehensive statistics and insights]
```

## Creating Custom Commands

Add new commands by creating `.md` files in this directory:

```bash
# Create new command
touch .claude/commands/my-command.md

# Edit with your favorite editor
code .claude/commands/my-command.md
```

**Template:**
```markdown
# Command Title

Description

## Instructions
1. Step 1
2. Step 2

## Code Examples
` ``python
# Example
` ``
```

## Tips

- Commands are loaded when Claude Code starts
- Restart Claude Code after adding new commands
- Keep command names short and descriptive
- Use hyphens for multi-word commands (e.g., `search-notes`)
- Document parameters and examples in the markdown

## MCP Integration

These slash commands work great with the MCP server!

**MCP Tools Available:**
- `search_notes` - Real-time database search
- `create_note` - Create notes via MCP
- `get_note` - Retrieve note details
- `get_vault_stats` - Get analytics
- `get_tags` - List all tags
- `get_recent_notes` - Recent activity

See `MCP_SETUP.md` for configuration.

## Troubleshooting

**Command not found?**
- Check file is in `.claude/commands/`
- Verify `.md` extension
- Restart Claude Code

**Command not working?**
- Check markdown syntax
- Ensure code blocks are properly formatted
- Look for typos in command name

## Contributing

Feel free to create custom commands for your workflow! Share useful ones with the team.
