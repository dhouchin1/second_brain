# Normalize tags into brain.tags and brain.file_tags from frontmatter 'tags'

ATTACH DATABASE 'brain.db' AS brain;

-- Extract candidate tags by splitting frontmatter 'tags' on commas/semicolons
WITH RECURSIVE raw_tags AS (
  SELECT file_id,
         lower(trim(replace(value, ';', ','))) AS val
  FROM brain.frontmatter
  WHERE key = 'tags' AND value IS NOT NULL AND trim(value) <> ''
),
split AS (
  SELECT file_id, val || ',' AS s, 1 AS pos FROM raw_tags
  UNION ALL
  SELECT file_id,
         s,
         pos + instr(substr(s, pos), ',') + 0 AS pos
  FROM split
  WHERE instr(substr(s, pos), ',') > 0
),
tokens AS (
  SELECT file_id,
         trim(replace(replace(substr(s, prev_pos, pos - prev_pos), '"',''), '\n', ' ')) AS token
  FROM (
    SELECT file_id, s, pos,
           lag(pos, 1, 1) OVER (PARTITION BY file_id, s ORDER BY pos) AS prev_pos
    FROM split
  ) t
  WHERE pos > prev_pos
),
normalized AS (
  SELECT file_id,
         trim(trim(replace(replace(replace(token, '[',''), ']',''), '#','')), ' ') AS tag
  FROM tokens
)
-- Insert unique tags
INSERT OR IGNORE INTO brain.tags(name)
SELECT DISTINCT tag FROM normalized WHERE tag <> '';

-- Link files to tags (recompute CTE scope)
WITH RECURSIVE raw_tags AS (
  SELECT file_id,
         lower(trim(replace(value, ';', ','))) AS val
  FROM brain.frontmatter
  WHERE key = 'tags' AND value IS NOT NULL AND trim(value) <> ''
),
split AS (
  SELECT file_id, val || ',' AS s, 1 AS pos FROM raw_tags
  UNION ALL
  SELECT file_id,
         s,
         pos + instr(substr(s, pos), ',') + 0 AS pos
  FROM split
  WHERE instr(substr(s, pos), ',') > 0
),
tokens AS (
  SELECT file_id,
         trim(replace(replace(substr(s, prev_pos, pos - prev_pos), '"',''), '\n', ' ')) AS token
  FROM (
    SELECT file_id, s, pos,
           lag(pos, 1, 1) OVER (PARTITION BY file_id, s ORDER BY pos) AS prev_pos
    FROM split
  ) t
  WHERE pos > prev_pos
),
normalized AS (
  SELECT file_id,
         trim(trim(replace(replace(replace(token, '[',''), ']',''), '#','')), ' ') AS tag
  FROM tokens
)
INSERT OR IGNORE INTO brain.file_tags(file_id, tag_id)
SELECT n.file_id, t.id
FROM (
  SELECT DISTINCT file_id, tag FROM normalized WHERE tag <> ''
) AS n
JOIN brain.tags t ON t.name = n.tag;

DETACH DATABASE brain;
