# Cleanup tag names by normalizing punctuation/case and consolidating duplicates

ATTACH DATABASE 'brain.db' AS brain;

-- Insert any missing normalized tags
WITH norm AS (
  SELECT id AS id_old,
         name AS name_old,
         lower(trim(replace(replace(replace(replace(name,'[',''),']',''), '#',''), '"',''), ' .,:;"')) AS name_new
  FROM brain.tags
)
INSERT OR IGNORE INTO brain.tags(name)
SELECT DISTINCT name_new FROM norm WHERE name_new <> '';

-- Remap file_tags to normalized tag ids
WITH norm AS (
  SELECT id AS id_old,
         lower(trim(replace(replace(replace(replace(name,'[',''),']',''), '#',''), '"',''), ' .,:;"')) AS name_new
  FROM brain.tags
),
map AS (
  SELECT n.id_old, t2.id AS id_new
  FROM norm n
  JOIN brain.tags t2 ON t2.name = n.name_new
)
UPDATE brain.file_tags
SET tag_id = (
  SELECT id_new FROM map WHERE map.id_old = brain.file_tags.tag_id
)
WHERE EXISTS (SELECT 1 FROM map WHERE map.id_old = brain.file_tags.tag_id)
  AND tag_id <> (
    SELECT id_new FROM map WHERE map.id_old = brain.file_tags.tag_id
  );

-- Delete duplicate/unnormalized tags that now have mappings
DELETE FROM brain.tags
WHERE id IN (
  SELECT id_old FROM (
    SELECT t.id AS id_old,
           lower(trim(replace(replace(replace(replace(t.name,'[',''),']',''), '#',''), '"',''), ' .,:;"')) AS name_new
    FROM brain.tags t
  ) d
  WHERE EXISTS (SELECT 1 FROM brain.tags t2 WHERE t2.name = d.name_new AND t2.id <> d.id_old)
);

DETACH DATABASE brain;
