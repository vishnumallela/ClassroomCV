-- English for every sentence that is not already English, so the numbers on
-- the page can be read by someone who reads neither Hindi nor Devanagari.
-- The transcriber writes much of the teacher's ENGLISH in Devanagari script,
-- so this is as often a de-transliteration as a translation. Filled by the
-- audio job through a language model; null when the sentence was already in
-- Latin script or the pass has not run. Kept per row so a re-run re-uses it
-- rather than paying to translate the same sentence twice.
ALTER TABLE "utterances" ADD COLUMN IF NOT EXISTS "text_en" text;
