# Teacher measurements — what the system produces

Every number the system reports about a lesson, and what each one needs before
it can exist. One requirement, one number, one way to compute it.

Source rubric: `Domain A.xlsx`, sheet "Domain A" — three parameters (A1 Timely
Start & Finish, A2 Efficient Routines & Behaviour, A3 Learning-Time), each with
its own SYSTEM MEASURES list. This document is the scoped subset actually being
built. It measures **the teacher, and only the teacher.**

---

## The eight things being measured

| What was asked for | Requirements |
| --- | --- |
| When the teacher arrived, and how late | R1, R2 |
| When the teacher left, and whether that was early | R3, R4 |
| When the lesson started, how long it ran, when it ended | R7, R8, R9, R10 |
| Whether the lesson fit inside the scheduled period | R11, R12 |
| How the lesson ended, or whether it continues next time | R13, R14, R15, R16 |
| How often the teacher raised her voice | R17 |
| How often the lesson drifted off topic | R19 |
| How many questions the teacher asked | R20 |
| How many languages the teacher used | R21 |

---

## First: two pairs that are not the same number

The rubric is strict about this and it is the easiest thing to get wrong.

**Arriving is not starting.** A teacher can be in the room ten minutes early
and still start late. R1 is when she walked in. R7 is when she set the first
learning task. The rubric grades R7.

**Leaving is not ending.** R3 is when she walked out. R9 is when teaching
stopped. A lesson can end ten minutes before anyone leaves the room.

---

## Facts someone has to type in

None of these are detected. Six requirements are impossible without them.

| ID | Input | Unlocks |
| --- | --- | --- |
| **P1** | Scheduled start time | R2, R8, R11, R12 |
| **P2** | Scheduled end time | R4, R5, R11, R12, R16 |
| **P3** | Subject and year group | context for R19, R20 |
| **P4** | Room type — classroom, lab, PE, library, practical | the reliability caveat |
| **P5** | Whether the same class has a following period | R14 |

P1 and P2 are the cheapest work in this document: one database column, no
machine learning, and four punctuality numbers appear immediately.

---

## Group A — Attendance

| ID | The number | Sensor | How | Status |
| --- | --- | --- | --- | --- |
| **R1** | **Arrival time** — the first moment the teacher is in the room | Camera | Start of the first presence interval | **Built** |
| **R2** | **Arrival against the bell** — minutes early or late | Camera + P1 | R1 − P1 | Needs timetable |
| **R3** | **Departure time** — the last moment she is in the room | Camera | End of the last presence interval, or the final door exit | **Built** |
| **R4** | **Departure against the bell** — minutes early or late | Camera + P2 | R3 − P2 | Needs timetable |
| **R5** | **Time in the room** — total present, and as a share of the scheduled period | Camera + P1/P2 | Sum of presence intervals | **Built**; the share needs the timetable |
| **R6** | **Mid-lesson absences** — how many times she left and for how long | Camera | Gaps between presence intervals, matched to door crossings | **Built** |

---

## Group B — The lesson

| ID | The number | Sensor | How | Status |
| --- | --- | --- | --- | --- |
| **R7** | **Lesson start** — when the first learning task was set | Mic + Camera | First teacher utterance that launches a task (starter, do-now, retrieval, discussion), corroborated by the first board interaction | **Provisional** (2026-09-04): phrase patterns in both scripts (`lib/phrases.ts`), board corroboration from the video (`lib/lesson-arc.ts`) |
| **R8** | **Start delay** — minutes from the bell to the first task | Mic + P1 | R7 − P1 | **Provisional** (2026-09-04), from R7 and the timetable |
| **R9** | **Lesson end** — when teaching stopped | Mic + Camera | Last teaching utterance, corroborated by the teacher leaving the board / the room | **Provisional** (2026-09-04): the later of the last teaching sentence and the last board interaction, capped at her departure |
| **R10** | **Lesson duration** | Mic + Camera | R9 − R7 | **Provisional** (2026-09-04) |
| **R11** | **Did the lesson fit the period?** | Mic + Camera + P1/P2 | Whether [R7, R9] sits inside [P1, P2] | **Provisional** (2026-09-04), one-minute tolerance |
| **R12** | **Overrun or underrun** — minutes past the bell, or minutes of the period left unused | Mic + Camera + P1/P2 | (R9 − P2) and (P2 − R9) | **Provisional** (2026-09-04) |

R8 is the number that grades punctuality. Everything else in this group is
reported alongside it so the figure can be read honestly — a lesson that
started on time and ended fifteen minutes early is not the same as one that ran
the full period.

**Read off the first real lesson (2026-09-04, `760713c7`):** start 09:54:49,
4.8 min after the bell — the first task-setting sentence ("take out your
literacy companion") with the first board interaction 34 s before it; end
10:34:53, on the bell; 40.1 min taught; the lesson fit the period. Every one
of these is PROVISIONAL: the sentence is found by phrase (`lib/phrases.ts`),
shown beside the number, and clickable to the video. Two limits met on the
way: "keep your almanac on the table" matched the pack-up cue (now the cue
needs the object put away and is only looked for in the last fifteen minutes),
and "teaching sentence" is by exclusion (not procedure, attention, pack-up,
homework or continuation), so the last one can be a behaviour remark. The
labelling pass replaces both.

---

## Group C — How the lesson ended

| ID | The number | Sensor | How | Status |
| --- | --- | --- | --- | --- |
| **R13** | **Closure, and its type** — review, reflection, exit question, summary, or none | Mic | Teacher utterances near the end labelled as closing | **Provisional** (2026-09-04): review / reflection / exit question / summary by phrase in the last five minutes; "none" is a reported outcome |
| **R14** | **Continuation** — did she say the lesson carries on next time | Mic + P5 | An explicit statement that the topic continues, e.g. "we'll finish this next class" | **Provisional** (2026-09-04), by phrase |
| **R15** | **Homework set** — whether homework was given, and when | Mic | Teacher utterance assigning work beyond the lesson | **Provisional** (2026-09-04), by phrase incl. "होमवर्क", "ब्रिंग इट टुमारो" |
| **R16** | **Pack-up instruction** — when the class was told to pack up, against the bell | Mic + P2 | First pack-up instruction, minus P2 | **Provisional** (2026-09-04): imperative phrases only, looked for in the last fifteen minutes |

R13 and R14 answer the same question from two directions: a lesson can end
properly with a summary, end properly by being explicitly carried over, or just
stop. All three are different outcomes and the report should name which.

---

## Group D — Voice and delivery

| ID | The number | Sensor | How | Status |
| --- | --- | --- | --- | --- |
| **R17** | **Raised-voice events** — count, and rate per ten minutes | Mic | Loudness of the teacher's own speech against her own rolling baseline, sustained past a minimum duration | **Built** (2026-09-04): `lib/loudness.ts` — ffmpeg RMS per 0.5 s folded onto each sentence and stored; an event is 6 dB over her median for ≥ 1.5 s, episodes merged within 5 s |
| **R18** | **Attention requests** — how many times she called for the class's attention | Mic | Teacher utterances flagged as an attention cue | **Provisional** (2026-09-04), by phrase in both scripts |
| **R19** | **Off-lesson drift** — how many episodes, and total minutes | Mic | Runs of teacher speech labelled as unrelated to the lesson | **Provisional** (2026-09-04): runs of administrative talk (notebooks, planners, signatures, fees) stand in |
| **R20** | **Questions asked** — count and rate, split into questions put to the class and rhetorical check-ins | Mic | Teacher utterances labelled as asking | **Provisional** (2026-09-04): question marks on the teacher's sentences, with check-ins set aside by a word list incl. tagged-on "…, ओके?"; reported as provisional until the labelling pass |
| **R21** | **Languages used** — which, how many, the share of speech in each, and switches per minute | Mic | Per-utterance language, normalised before counting | **Built** (2026-09-04) from each sentence's script — an upper bound on Hindi, since the transcriber writes some English in Devanagari |

Three notes that decide whether these numbers are worth anything:

- **R17 is loudness, not words.** Transcription returns no volume. This is a
  separate pass over the audio waveform, and it is the only requirement here
  that does not read the transcript.
- **R20 must separate real questions from filler.** "ठीक है?", "समझे?" and
  "right?" are punctuation, not questions. Counting question marks overstates
  this number several times over.
- **R21 is not in the rubric.** It is an addition, and a useful one in a
  classroom that code-switches. Report the set of languages, the share of each,
  and the switch rate — the last is the interesting one.

**What the first real lesson taught (2026-09-04, `760713c7`, 45 min).** The
transcript is stored as SENTENCES cut from the words (`lib/segment.ts`), not
as diarizer turns — a turn ran 6.5 minutes and hid 98 pauses. The teacher's
voice is the diarized speaker that carries the speech while the video says
she is in the room (`lib/voice.ts`): speaker B, 88% of in-presence speech,
which is also what separates her from the period-2 teacher's opening minutes.
Read off that lesson: teacher talk 59%, others 14%, silence 27%; longest
stretch 3:21; 133 words/min; 70 teacher turns; ~50 questions to the class
(provisional); Hindi script 72% / English 26% / mixed 3%, 1.6 switches a
minute; 73% of the recording transcribed at 0.89 confidence. Two limits
found there: the transcriber renders some English in Devanagari ("आई विल साइन
एंड देन रिटर्न"), so script is an upper bound on Hindi; and question marks
alone overstate R20 by roughly a third even after tag questions are set aside.
R7-R16 and R17-R19 need the labelling pass (an LLM key) and the loudness
pass; the card names them as pending rather than showing zeros.

---

## Group E — Trust

| ID | The number | Sensor | How | Status |
| --- | --- | --- | --- | --- |
| **R22** | **Observation coverage** — how much of the lesson the system could see and hear, and how confident it is | Camera + Mic | Coverage, continuity and detection confidence, per sensor | **Built** for video and audio (transcribed share, transcription confidence) |
| **R23** | **Not Observed** — the system says so rather than guessing | — | Any requirement below its evidence threshold reports Not Observed | **Built** (2026-09-04): every measurement carries observed / provisional / not observed with its reason; the Trust card lists all 23 |

---

## What has to be built

Five pieces of work. Each unlocks a fixed set of requirements, and they are
listed in the order that returns the most for the least.

| # | Work | Unlocks |
| --- | --- | --- |
| 1 | **Timetable fields** — scheduled start and end on every lesson | R2, R4, R5, R8, R11, R12, R16 |
| 2 | **Audio capture and transcription**, with the teacher's voice separated from everyone else's | R7, R9, and everything in Groups C and D |
| 3 | **Loudness pass** over the audio waveform | R17 |
| 4 | **Utterance labelling** — what each teacher utterance was doing | R13, R14, R15, R16, R18, R19, R20 |
| 5 | **Language detection and normalisation** | R21 |

Step 1 is a database column and a form field. Step 2 is the large one — thirteen
requirements sit behind it.

---

## Known measurement risks

Both found in testing, both affect numbers in Group D.

- **Student speech gets absorbed into the teacher's turns.** A single room
  microphone cannot reliably separate one teacher from thirty students, so
  anything measured as "the teacher's speech" is overstated. A clip-on
  microphone for the teacher plus a room microphone removes this entirely, and
  is the recommended recording setup.
- **The same word appears in two scripts.** In code-switched audio an English
  term can be transcribed in Latin or in Devanagari within the same lesson.
  Text must be normalised before anything counts occurrences, or R19, R20 and
  R21 all undercount.

---

## Deliberately out of scope

Not measured, and not to be inferred:

- **Anything about students** — whether they were working, out of their seats,
  packing up early, or settled. The system does not detect students.
- **Transitions between activities** — count, length or cost.
- **Settle-after-cue time** — it requires knowing whether the class went quiet
  *and* still, which needs the students.
- **Learning-time share** — a percentage of the period spent learning cannot be
  produced honestly without the above.
- **Off-task noise** — meaningless without knowing what the class was meant to
  be doing.
- **Lesson-plan phase context** — no lesson plan is required as an input.

Where the rubric grades one of these, the system reports **Not Observed** for
that band rather than a number it cannot stand behind.

---

## Rules that apply to every number

From the rubric's own closing note:

1. Never judge from a single moment — a signal must persist to count.
2. Anchor start and finish to the timetable, never to the length of the recording.
3. Where the observation is insufficient, state **Not Observed**.
4. Labs, PE, library and practical lessons have different movement and noise
   norms; lessons with no audible bell and double periods break the timing
   assumptions. In these cases the reading is declared unreliable.
