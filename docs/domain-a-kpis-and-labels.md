# Domain A — KPIs and labels

What the system can tell a teacher about their lesson, what has to be built
before each number is possible, and what has to be labelled to get there.

## Where things stand

The system watches video only. It can see the teacher, the door and the screen,
and it can tell when the teacher is pointing or writing. It cannot hear
anything, and it cannot see students.

Twelve useful numbers work today, but none of them is a grade. The rubric's
three grades all need measurements the system cannot yet make.

**Scope.** This covers **teacher analysis only**. Nothing here measures
students, and no measurement that depends on detecting them is included.

This document lists forty-six numbers, grouped by what has to exist before each
one becomes possible — then the labelling work each group depends on. The groups
run in order, and none can be skipped.

## The three numbers that matter most

One for each thing the rubric grades.

### 30. Start delay — Punctuality

> How many minutes passed between the bell and the teacher setting the first learning task.

**What it needs.** The instruction heard on the microphone, and the timetable to measure it against.

**Why it is hard.** Being in the room is not the same as having started. A teacher can arrive ten minutes early and still start late, so the measurement has to find the moment a task was actually set, not the moment the teacher walked in.

### 31. Settle-after-cue time — Behaviour

> After the teacher asks for quiet, how many seconds until the room is quiet again.

**What it needs.** The cue and the noise level that follows it, both from the microphone, plus the lesson plan to know what quiet should sound like.

**Why it is hard.** The cue has to be told apart from ordinary instruction, and “quiet” is not a fixed volume — silent practice and group work have completely different baselines.

### 32. Learning-time share — Learning time

> What percentage of the scheduled period the teacher spent on learning activity.

**What it needs.** The lesson marked up into phases, the teacher's speech, and the timetable.

**Why it is hard.** Quiet is ambiguous. A class working silently and a teacher fighting the projector sound the same, so the lesson plan and what the teacher is doing have to settle which one it was.

---

# Part one — all forty-six numbers

**Every number here is about the teacher, and only the teacher.** Nothing is
measured about students, individually or as a group, and nothing here depends on
the system being able to see them. Where the rubric describes what students did,
this plan reads it from the teacher's side instead — what the teacher said, set
up, or did about it.

## Group 1 — Live today

*Needs nothing new. These already work.*

Worth being honest about what these are: they describe activity, not achievement. None of them produces a grade, and none is anchored to a timetable.

1. **Teacher presence** — How much of the recording the teacher was in the room for.
2. **Time at board** — How long the teacher spent at the board, in minutes and as a share of the lesson.
3. **Pointing** — How long the teacher spent pointing at the board or screen.
4. **Writing** — How long the teacher spent writing on the board.
5. **Board sessions** — How many separate times the teacher went to the board, and how long each lasted.
6. **Teacher entries** — How many times the teacher came into the room.
7. **Teacher exits** — How many times the teacher left the room.
8. **Room covered** — How much of the floor the teacher's path touched.
9. **Movement spread** — Whether the teacher's time was spread across the room or concentrated in a few spots.
10. **Anchor spot** — How much of the lesson the teacher spent in their single most-used position.
11. **Movement style** — Front-of-room presenter, circulating supervisor, or balanced.
12. **Observation coverage** — How much of the lesson the system could actually see, and how confident it is.

> On movement style. It describes where the teacher taught from, not how well. It must never be shown as a quality score — it is not one of the rubric's measures.

## Group 2 — Add the timetable

*Needs a scheduled start and end time, typed in. No machine learning at all.*

The cheapest unlock in this document. Teacher presence already works; a scheduled start and end turn those floating numbers into punctuality measures.

13. **Teacher arrival, against the bell** — How many minutes before or after the scheduled start the teacher was in the room.
14. **Teacher departure, against the bell** — How many minutes before or after the scheduled end the teacher left.
15. **Presence against scheduled time** — Teacher presence measured against the period the timetable gave, not the length of the recording.
16. **Recording coverage of the period** — Whether the camera actually captured the whole lesson.

> This does not give punctuality its grade — the rubric grades when learning started, not when the teacher arrived. But these four are honest, useful, and cost one database column.

## Group 3 — Add sound

*Needs a microphone, and a way to tell the teacher's voice from everyone else's.*

The largest single unlock, and the point at which behaviour becomes gradeable at all. Four of the seven things the rubric asks for in that parameter are things you hear, not things you see.

17. **Attention requests** — How many times the teacher had to ask for the class's attention.
18. **Raised-voice events** — How often the teacher raised their voice — a count and a rate per ten minutes.
19. **Behaviour talk vs teaching talk** — What share of the teacher's talking was about behaviour rather than the subject.
20. **Teacher talk share** — How much of the lesson's talking time was the teacher's.
21. **Repeated instructions** — How often the teacher had to give the same procedural instruction again.
22. **Redirections** — How many corrections the teacher made, and how long each one took.
23. **Off-task noise share** — What share of the lesson had off-task noise running underneath it.
24. **Unrelated talk** — Minutes of talk that had nothing to do with the lesson.
25. **Questions the teacher asked** — How many questions, and how often.
26. **First instruction** — The moment the teacher gave the instruction that launches a task.
27. **Attempts to begin** — How many times the teacher tried to start before it worked.
28. **Pack-up instruction** — When the teacher told the class to pack up, against the bell.
29. **Closure present, and its type** — Whether the lesson had a proper ending, and what kind.

> #23 comes with a condition. Off-task noise means nothing without knowing what the class was supposed to be doing. Shipped without that, it punishes a well-run group-work lesson for being loud. Ship it with the lesson plan, or not at all.

## Group 4 — Read picture and sound together

*Needs the camera and microphone read as one, plus the lesson plan.*

Where the rubric's own measures live, and the only place the three grades can be produced.

33. **Lesson time breakdown** — Where the period went: teaching, settling, changes of activity, behaviour, and time lost to equipment.
34. **Changes of activity** — How many there were, how long each took from the instruction to teaching resuming, and what they cost in total.
35. **Routine reliance** — What share of activity changes needed only one instruction rather than being talked through step by step.
36. **Intervention timing** — Whether the teacher stepped in as the noise started to rise, or only after it had built.
37. **Teaching stoppages for behaviour** — How often teaching stopped completely, and for how long.
38. **Dead and tech time** — Minutes lost to equipment and setup.
39. **Activity mix the teacher ran** — Minutes spent explaining, questioning, discussing, setting practice, checking, giving feedback and closing.
40. **The three grades** — Emerging, Developing, Competent — or Not Observed — for each parameter.

> The three headline numbers — start delay, settle-after-cue time and learning-time share — also belong to this group. They are set out at the top of this document.

## Trend — Across lessons

*Any group's numbers, rolled up over time.*

The single-lesson numbers matter far less to a teacher than the direction they are moving in.

41. **Lessons observed, hours analysed** — How much has been looked at. Already live.
42. **Start delay over time** — Is the teacher starting lessons faster than they were?
43. **Learning-time share over time** — Is more of the period going to teaching?
44. **Settle time over time** — Is the class responding to cues more quickly?
45. **Raised-voice rate over time** — Is the teacher needing to raise their voice less?
46. **Grade progression** — How the three parameters have moved across the term.

## Numbers we should refuse to show

- **A learning-time percentage from video alone.** It would be a guess presented as a measurement, because quiet is unreadable without sound.
- **Off-task noise without knowing the lesson plan.** It punishes planned group work for being loud.
- **Any grade when the camera could not see much.** The rubric asks for “Not Observed” instead, and that is the honest answer.
- **Anything about students.** Out of scope entirely. The system does not see them, and no number here should imply it does.
- **Movement style as a quality score.** It says where a teacher taught from, not how well they taught.
- **A grade for a lab, PE or practical lesson with no caveat.** Movement and noise mean different things there, and the system has to say so.

---

# Part two — what has to be labelled

## How the dataset works today

One model, trained to recognise five things, in this fixed order:

```
0 Door · 1 Screen · 2 Teacher · 3 pointing · 4 writing
```

1. **That order is a contract.** The service checks it every time the model
   loads. Anything new has to be added at the end, from 5 onwards — inserting
   one in the middle renumbers the rest, and the system starts reporting the
   door as the teacher.
2. **“Screen” is what the product calls the board.** Two names for one thing
   already meet in a single place in the code. Do not create a second pair.
3. **Only the teacher is saved.** The database keeps teacher detections and
   nothing else — not even the pointing and writing boxes.

## New things to label — the teacher

The model already treats teacher actions as things in their own right: pointing
and writing are two of the five. Three more teacher measurements need the same
treatment. Without them, those numbers fall back on guessing from position,
which will be far less reliable.

| id | Name | Effort | Why it is needed |
| --- | --- | --- | --- |
| 5 | `teacher_at_computer` | cheap | Time lost to equipment is *the teacher operating it*, not a computer being visible somewhere in the room. An action class measures the thing the rubric actually grades. |
| 6 | `teacher_crouching` | cheap | The rubric names stopping at a desk as a mark of a well-run room. Position alone cannot tell walking past one from stopping at it. Posture can. |
| 7 | `teacher_hand_raised` | cheap | Attention cues are not only spoken — a raised hand is a cue. Lets the system see one without relying entirely on the microphone. |

All three are cheap to label: one box per frame, on someone the model already
finds reliably.

## New things to label — the room

| id | Name | Effort | Why it is needed |
| --- | --- | --- | --- |
| 8 | `materials` | moderate | Books, worksheets and equipment out and ready. Whether the teacher had a task prepared and waiting, which is what the rubric asks about the start of a lesson. |

## A naming decision to take at the same time

`pointing` and `writing` are the **teacher's** actions, but neither name says
so. Every new class above is explicitly prefixed `teacher_`, which leaves two of
the five original names as the odd ones out. Two things make this cheap to fix
now and awkward to fix later:

1. **The model is being retrained anyway.** Adding anything new requires it,
   and retraining rewrites the name list. The rename costs nothing extra.
2. **Saved records do not use these names.** The event log writes its own text,
   so renaming what the model calls a class does not touch a stored row.

**Recommendation:** rename them to `teacher_pointing` and `teacher_writing` in
the same retrain that adds the new classes. Leave their positions alone — the
order is the contract.

## Three other kinds of labelling

### What is on the screen

Not a new thing to find, just a judgement about something the system already
finds: is the screen showing lesson content, a desktop or menu, nothing at all,
or something still loading? Plus whether the board has been written on. This is
what separates a task being ready from a projector being wrestled with, and it
is far cheaper than teaching the model to find something new.

### Sound

None of this exists in any form, and it is a separate job from the video.

- **Who is speaking** — the teacher, someone else, several people, or nobody. Needed to isolate the teacher's voice, which almost everything in group three depends on.
- **What the teacher meant** — instructing, explaining, asking, giving feedback, managing behaviour, going through procedure, off topic, or closing the lesson.
- **Whether the teacher's voice was raised**
- **Whether an utterance was a call for attention**
- **What the room sounds like** — silent, a working hum, or off-task chatter.
- **Whether a bell or chime sounded** — as a fallback where no timetable is available.

> **Only the teacher's speech needs interpreting.** Who is speaking and what the
> room sounds like have to be marked across the whole recording, but meaning,
> raised voice and attention cues apply to the teacher alone. That decision
> removes most of the transcription effort.

### The shape of the lesson

Marking up the lesson as a timeline: settling, starter, explanation, practice,
group work, changes of activity, behaviour, time lost to equipment, and the
closing. Plus the single moments — when the first task was set, where each
change of activity started and ended, where each cue was answered, and where the
closing ran.

This is the best value in the whole plan. A lesson has perhaps thirty segments
against ten thousand frames, so it is quick to mark, and it describes the lesson
*the teacher ran*.

### Facts to type in, with no labelling at all

- **The scheduled start and end time** — Punctuality cannot be graded without it, and learning time has no denominator without it.
- **Subject and year group** — for context.
- **The lesson plan** — which is what makes noise readable — silent work should be quiet, group work should not be.
- **The kind of room** — ordinary classroom, lab, PE, library, practical — because the norms differ.
- **Whether there is an audible bell, and whether this is a double period**

## What this scope deliberately leaves out

Three things the rubric asks about are only visible from the students' side:
whether the class was actually working, how many were out of their seats, and
whether the closing was drowned out by packing up. None of them is in this plan.

That is a real cost, and it shows up in two places. **Start delay** measures the
moment the teacher set a task, not the moment the class began working on it — a
teacher who gives a clear instruction that nobody acts on scores the same as one
whose class starts immediately. And **learning-time share** is built from what
the teacher was doing rather than what the room was doing, so it is an estimate
of the opportunity offered rather than the work done.

Both are still worth reporting, as long as they are labelled for what they
measure. Extending to the students' side is a separate decision, with its own
consequences for storage, for whether the system has to follow individuals
around, and for holding recordings of identifiable children.

## The order to do it in

By value returned for effort spent.

1. **Type in the timetable and lesson details.** No labelling at all, and four real punctuality numbers appear immediately.
2. **Mark up the shape of each lesson.** Quick per lesson, and it is the skeleton learning time hangs from. It also supplies the lesson context every sound measurement later depends on.
3. **The three teacher classes.** The cheapest labelling in the plan — one box per frame, on someone the model already finds reliably. Do the renaming in this same retrain.
4. **Sound.** The largest single effort by a wide margin, and the one that makes behaviour gradeable at all. Thirteen numbers depend on it, and so do all three headline numbers.
5. **Screen and board states, and `materials`.** Completes the picture of time lost to equipment, and whether a task was ready at the start.

The first three steps are worth finishing before anything else. They cost the
least, they produce real numbers, and step two makes all later model work
measurable. Step four is where the real effort sits — everything the rubric
grades about behaviour is on the far side of it.

---

## Appendix — where each number comes from

For anyone cross-checking against `docs/domain-a-requirements.md`.

| # | Number | Requirement |
| --- | --- | --- |
| 1 | Teacher presence | A2-1 |
| 2 | Time at board | A2-2 |
| 3 | Pointing | A2-2 |
| 4 | Writing | A2-2 |
| 5 | Board sessions | A2-2 |
| 6 | Teacher entries | A1-9 |
| 7 | Teacher exits | A1-10 |
| 8 | Room covered | A2-1 |
| 9 | Movement spread | A2-1 |
| 10 | Anchor spot | A2-1 |
| 11 | Movement style | A2-1 |
| 12 | Observation coverage | X-4, X-5 |
| 13 | Teacher arrival, against the bell | A1-1, X-1 |
| 14 | Teacher departure, against the bell | A1-10, X-1 |
| 15 | Presence against scheduled time | A1, A3 |
| 16 | Recording coverage of the period | X-1, X-5 |
| 17 | Attention requests | A2-8 |
| 18 | Raised-voice events | A2-10 |
| 19 | Behaviour talk vs teaching talk | A2-11 |
| 20 | Teacher talk share | A2-11, A3-7 |
| 21 | Repeated instructions | A2-12 |
| 22 | Redirections | A2-13 |
| 23 | Off-task noise share | A2-15 |
| 24 | Unrelated talk | A3-5 |
| 25 | Questions the teacher asked | A3-7 |
| 26 | First instruction | A1-3 |
| 27 | Attempts to begin | A1-4 |
| 28 | Pack-up instruction | A1-13 |
| 29 | Closure present, and its type | A1-11 |
| 30 | Start delay | A1-3, A1-7 |
| 31 | Settle-after-cue time | A2-8, A2-9 |
| 32 | Learning-time share | A3-1, A3-2 |
| 33 | Lesson time breakdown | A3-1 |
| 34 | Changes of activity | A2-3, A2-4, A2-5 |
| 35 | Routine reliance | A2-6 |
| 36 | Intervention timing | A2-19 |
| 37 | Teaching stoppages for behaviour | A2-18 |
| 38 | Dead and tech time | A3-3 |
| 39 | Activity mix the teacher ran | A3-7 |
| 40 | The three grades | all |
| 41 | Lessons observed, hours analysed | live |
| 42 | Start delay over time | A1 |
| 43 | Learning-time share over time | A3 |
| 44 | Settle time over time | A2 |
| 45 | Raised-voice rate over time | A2 |
| 46 | Grade progression | all |
