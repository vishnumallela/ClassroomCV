# Domain A — what the system must observe, and with which sensor

Source: `Domain A.xlsx`, sheet "Domain A". A three-band rubric — **Emerging /
Developing / Competent** — over three parameters. This document turns that
rubric into a list of things the system has to be able to work out, says which
sensor each one needs, and records what already works today.

The rubric's own framing:

> Maximising students' opportunity to learn. A well managed classroom is a
> floor, not a ceiling (behaviour management ~0.35 in Visible Learning) —
> scored as the base a lesson stands on.
>
> T: = teacher actions, S: = student actions; as seen/heard in the recording.

The three parameters in scope:

| Rubric | Full name | What it really asks |
| --- | --- | --- |
| A1 | Timely Start & Finish | Did the lesson start and stop when it was supposed to? |
| A2 | Efficient Routines & Behaviour | When the lesson was interrupted, how quickly did it recover? |
| A3 | Learning-Time | What share of the period was actually spent learning? |

---

## How to read this document

Every requirement is written as **a question the system has to answer**,
followed by *how* it would answer it. Two labels sit alongside each one.

### The sensor label — what it takes to answer the question

| Label | Meaning |
| --- | --- |
| **Camera** | The video alone is enough. The system can see it happen. |
| **Mic** | The audio alone is enough. The system can hear it happen. |
| **Camera + Mic** | Neither one is enough on its own. Video and audio have to be read together, or the answer will be wrong. |
| **Provided** | Not something to detect at all. Information somebody has to give the system, like the timetable. |

The **Camera + Mic** label is the important one. It does not mean "nice to have
both". It means a system using only one sensor will confidently produce the
wrong band. Each parameter below has a worked example showing exactly how.

### The status label — what exists today

| Label | Meaning |
| --- | --- |
| **Built** | Working now in the current pipeline. |
| **Partly built** | A related signal exists, but not the measurement the rubric actually asks for. |
| **Not built** | Nothing exists yet. |

---

## Where the system stands today

The current pipeline is **video only**. Its detector recognises exactly five
kinds of thing: `Door`, `Screen`, `Teacher`, `pointing`, `writing`. From those
it produces: when the teacher was present, when they came in and went out
through the door, how long they spent at the board, when they were pointing or
writing, a heatmap of where they stood, how much they moved, and a confidence
score for how well it could see.

Two absences decide almost everything in this document:

- **There is no audio at all.** No recording, no transcription, no way of
  telling one voice from another.
- **The detector cannot see students.** It only knows what a teacher looks
  like. Every line in the rubric beginning `S:` is currently unmeasurable.

Counting the 50 requirements:

| Sensor | Total | Built | Partly built | Not built |
| --- | --- | --- | --- | --- |
| Camera | 13 | 3 | 4 | 6 |
| Mic | 13 | 0 | 0 | 13 |
| Camera + Mic | 20 | 0 | 0 | 20 |
| Provided | 4 | 0 | 0 | 4 |

Which gives three conclusions:

1. **Audio is the bottleneck.** Nothing that needs a microphone exists, and the
   twenty "Camera + Mic" requirements are blocked behind it too. That is 33 of
   50 requirements waiting on one missing capability. Parameter A2 cannot be
   scored at all without it.
2. **Teaching the detector to see students unlocks the video half.** Six of the
   camera requirements are missing for that one reason and no other. There is
   nothing else difficult about them.
3. **Two of the gaps need no technology whatsoever.** The timetable and the
   lesson plan are facts somebody types in. A1 cannot be scored without the
   timetable, because "late" has no meaning without a scheduled start.

---

# A1 — Timely Start & Finish

> Scheduled learning time is protected by starting and finishing the lesson on
> time. Judged against the timetable.

**In plain terms:** did teaching begin promptly, and did the lesson get a
proper ending instead of fizzling out into packing up?

The critical thing to understand about A1 is that **it does not grade when the
teacher walked in.** It grades when *learning started*. A teacher can be in the
room ten minutes early and still score Emerging. All three bands are defined by
the delay before the first learning activity:

| Band | Start delay |
| --- | --- |
| Emerging | around 10 minutes late |
| Developing | around 5–10 minutes late |
| Competent | within about 5 minutes |

### What the rubric says

| Emerging | Developing | Competent |
| --- | --- | --- |
| **START** — T: Arrives late, still settling students, organising materials or in informal talk well past the scheduled start; makes repeated attempts to begin. First learning activity (starter, hook or recall) starts approximately 10 min late. S: Stay unsettled with nothing to begin.<br><br>**FINISH** — T: The closure of the lesson is displaced by packing up, or the lesson overruns because planned teaching isn't finished. S: Pack up well before the lesson ends, or are still working past it. | **START** — T: Arrives on time, begins organising around the scheduled start but needs repeated prompting to settle the class. First learning activity starts approximately 5–10 min late. S: Seated but idle for several minutes before working.<br><br>**FINISH** — T: Finishes close to time, but the plenary is shortened, rushed or interrupted by packing up. S: Start packing up or disengage as the plenary begins, so the closing activity noise is similar to settling-down noise. | **START** — T: A task is ready (starter / do-now / retrieval / discussion) and instructions are brief; first learning activity starts within ~5 min of the scheduled start. S: Begin the available task with materials to hand.<br><br>**FINISH** — T: Ends close to the scheduled time with the planned closure (review, reflection, exit question or summary) actually taking place rather than displaced by packing up. S: Stay engaged in the closing task (review, reflection, exit question) up to the scheduled end; packing up happens only after it. |

The rubric also lists the numbers it wants out: start delay, end versus
scheduled end, whether a proper close happened or was displaced by packing up,
and mid-lesson late arrivals.

### The start of the lesson

| ID | Question the system must answer | Sensor | Status |
| --- | --- | --- | --- |
| **A1-1** | **Was the teacher in the room when the lesson was due to start?**<br>Compare the first frame the teacher appears in against the scheduled start time. | Camera | **Built** |
| **A1-2** | **Once in the room, was the teacher getting ready to teach — or sorting out materials and chatting?**<br>The rubric treats these very differently, but they look almost identical on video. What separates them is what is being said. | Camera + Mic | Not built |
| **A1-3** | **When did the teacher give the instruction that actually launches a task?**<br>Find the first instruction that sets a starter, do-now, retrieval activity or discussion going. | Mic | Not built |
| **A1-4** | **How many times did the teacher try to start before it worked?**<br>The rubric's Emerging band names "repeated attempts to begin" directly. | Mic | Not built |
| **A1-5** | **Was there a task waiting for students — on the board, on the screen, or on their desks?**<br>Competent requires a task to be *ready*, not just announced. | Camera | **Partly built** — the screen is detected, but nothing reads what is on it |
| **A1-6** | **When did learning actually begin?**<br>The moment a task is live *and* students are working on it. Every other A1 number is measured from here. | Camera + Mic | Not built |
| **A1-7** | **How many minutes late was that?**<br>Subtract the scheduled start from A1-6. This single number picks the band. | Provided + Camera + Mic | Not built — needs A1-6 and X-1 |
| **A1-8** | **Were students left sitting with nothing to do?**<br>The Emerging band's "stay unsettled with nothing to begin". | Camera | Not built — the detector cannot see students |
| **A1-9** | **Did anyone arrive late, after the lesson had already started?** | Camera | **Partly built** — the door is tracked, but only the teacher can be recognised passing through it |

### The end of the lesson

| ID | Question the system must answer | Sensor | Status |
| --- | --- | --- | --- |
| **A1-10** | **Did the teacher stay until the scheduled end?**<br>Flag an early departure. | Camera | **Built** |
| **A1-11** | **Was there a proper ending — a review, reflection, exit question or summary?**<br>The rubric calls this the closure or plenary. Recognising it means recognising what was said. | Mic | Not built |
| **A1-12** | **Did that ending actually take place, or was it swamped by packing up?**<br>The rubric's word is *displaced*. A closure that nobody was listening to does not count. | Camera | Not built — needs to see students |
| **A1-13** | **When were students told to pack up, compared to the bell?** | Mic | Not built |
| **A1-14** | **Did the lesson run over because the planned teaching wasn't finished?** | Camera + Mic | Not built |
| **A1-15** | **How many minutes early or late did the lesson end?** | Provided + Camera + Mic | Not built |

### Why A1-6 and A1-12 need both sensors

**Two lessons that a single sensor cannot tell apart.**

At 9:03 a teacher says, *"Right everyone, page forty, off you go."*

- In the first lesson, students open their books and start. Learning began at
  9:03 — three minutes late. **Competent.**
- In the second, the class carries on chatting. Nobody starts until 9:11 —
  eleven minutes late. **Emerging.**

A microphone hears identical audio in both. A camera sees a teacher at the
front and a seated class in both. Only the two together separate a Competent
lesson from an Emerging one — and the rubric's Developing band, *"seated but
idle for several minutes before working"*, describes precisely the case each
sensor alone misreads as fine.

The same trap sits at the end of the lesson, and the rubric says so itself:
*"the closing activity noise is similar to settling-down noise."* Audio cannot
separate a plenary from packing up. Video can — it can see whether students are
still working or already zipping up bags.

---

# A2 — Efficient Routines & Behaviour

> Transitions, routines, materials and behaviour are managed so interruptions
> to learning are brief and few.

**In plain terms:** the lesson will be interrupted — that is normal. A2 asks
how fast it recovers, and how much the teacher has to spend to make that
happen.

This is the most audio-dependent parameter in the domain. Four of the seven
measurements the rubric asks for are things you hear, not things you see, and
the band descriptions turn on **how often the teacher raises their voice**, **how
many prompts the class needs**, and **how quickly it goes quiet to a cue**. A
video-only system is close to blind here.

### What the rubric says

| Emerging | Developing | Competent |
| --- | --- | --- |
| T: Repeatedly calls for attention and re-states procedural instructions; raised voice is frequent; transitions are long or unclear; stops teaching repeatedly to deal with behaviour; redirection is inconsistent, delayed or drawn-out. When asked to settle, the class quietens slowly or only partly.<br><br>S: Frequently wait, search for materials or ask what to do; off-task noise is frequent or sustained; slow to return to task after redirection. | T: Announces and manages each transition personally, organises materials, and gives repeated reminders about expectations; tends to step in after off-task noise has risen rather than as it begins; raised voice is occasional. The class settles, but usually only after more than one prompt.<br><br>S: Some stay off-task during transitions or just after instructions, causing recurring small losses of time. | T: Relies on established routines — most changes of activity need one clear instruction or a familiar cue; addresses minor off-task behaviour early and briefly (a word, a pause, moving closer); raised voice is rare. The class goes quiet quickly to an established cue.<br><br>S: Begin, move between activities and access materials with little step-by-step direction; return to learning quickly; working noise fits the activity rather than being off-task. |

### Transitions and routines

A *transition* is any change from one activity to the next — book work to
discussion, sitting to grouping. The rubric cares about how long they take and
how much instruction they need.

| ID | Question the system must answer | Sensor | Status |
| --- | --- | --- | --- |
| **A2-1** | **Where was the teacher, and how much did they move around?** | Camera | **Built** — heatmap, path and movement score |
| **A2-2** | **How long did the teacher spend at the board, writing, or pointing?** | Camera | **Built** |
| **A2-3** | **When did the lesson change from one activity to another?**<br>Every other transition number depends on finding these boundaries first. | Camera + Mic | Not built |
| **A2-4** | **How long did each change take, from the instruction to students actually working again?** | Camera + Mic | Not built |
| **A2-5** | **How many changes were there, and what did they cost in total?** | Camera + Mic | Not built |
| **A2-6** | **Did the class move because of a routine they already know, or because the teacher walked them through it step by step?**<br>This is the exact line between Competent and Developing: one clear cue, versus announcing and managing each transition personally. | Camera + Mic | Not built |
| **A2-7** | **Were students waiting, hunting for equipment, or asking what to do?** | Camera | Not built — needs to see students |

### Attention and the teacher's voice

| ID | Question the system must answer | Sensor | Status |
| --- | --- | --- | --- |
| **A2-8** | **How many times did the teacher ask for the class's attention, and when?** | Mic | Not built |
| **A2-9** | **After the teacher asked for quiet, how many seconds until the class was actually settled?**<br>The single measurement that most cleanly separates the three bands. | Camera + Mic | Not built |
| **A2-10** | **How often did the teacher raise their voice?**<br>The rubric grades this directly: frequent, occasional, or rare. | Mic | Not built |
| **A2-11** | **How much of the teacher's talking was about behaviour rather than about the subject?** | Mic | Not built |
| **A2-12** | **Did the teacher have to repeat the same procedural instruction?** | Mic | Not built |
| **A2-13** | **When correcting a student, was it a quick word — or a drawn-out exchange?**<br>Competent is "a word, a pause, moving closer". Emerging is "inconsistent, delayed or drawn-out". | Mic | Not built |
| **A2-14** | **Did the class go quiet quickly, slowly, or only partly?**<br>"Only partly" is its own Emerging descriptor and needs to be distinguishable from "slowly". | Camera + Mic | Not built |

### Behaviour management

| ID | Question the system must answer | Sensor | Status |
| --- | --- | --- | --- |
| **A2-15** | **How much of the lesson had off-task noise running underneath it?** | Mic | Not built — and only meaningful alongside X-3 |
| **A2-16** | **How often were students out of their seats when they shouldn't be?** | Camera | Not built — needs to see students |
| **A2-17** | **Did the teacher handle off-task behaviour by moving closer to the student?**<br>The rubric names moving closer as a Competent technique, so it has to be recognised rather than missed. | Camera | Not built — needs to see students |
| **A2-18** | **How often did teaching stop completely to deal with behaviour, and for how long?** | Camera + Mic | Not built |
| **A2-19** | **Did the teacher step in as things started to slip, or only once the noise had built up?** | Camera + Mic | Not built |

### Why A2-9 and A2-19 are the hard ones

**A2-9 — what "settled" actually means.** The teacher says "quiet please" at
10:15.

- The camera says everyone was back in their seat by 10:15:04. Settled?
- The microphone says the talking carried on until 10:15:40. Not settled.

Both readings are true, and each on its own is wrong. A class can be silent and
still wandering about; it can be seated and still talking over the teacher.
"Settled" only holds when the noise **and** the movement have both come down,
so the measurement needs both sensors by definition.

**A2-19 — timing an intervention.** The rubric separates a teacher who steps in
"as it begins" from one who steps in "after off-task noise has risen". The
action is the same and the words may be identical. The only difference is
*where on the rising curve of disruption it lands*. So the system needs a
continuous picture of how off-task the room is getting — which takes both
sensors — plus the moment the teacher intervened, which is audio. No single
sensor can produce this.

---

# A3 — Learning-Time

> The share of scheduled time in which students have a clear learning purpose.
>
> Learning-time share = 1 − (settling + transitions + behaviour + dead/tech
> time) ÷ scheduled time.

**In plain terms:** out of the 50 minutes the timetable gave this lesson, how
many were actually spent learning?

A3 is a summary of the other two, and the rubric says as much: it notes the
score correlates with A1 and A2. Three of the four things subtracted — settling,
transitions, behaviour — are already being measured for A1 and A2. A3 only adds
two ideas of its own: time lost to technology, and talk that has nothing to do
with the lesson.

What A3 really demands is a **complete second-by-second account of the lesson**.
Every second has to be given exactly one label, with none left over, because the
score is a percentage of the whole period.

### What the rubric says

| Emerging | Developing | Competent |
| --- | --- | --- |
| Under ~50% of the period is learning time. Long stretches go to settling, behaviour, materials, avoidable technology/setup problems or unrelated talk. Many students are without a learning task for sustained periods. | ~50–75% is learning time. Most of the lesson is purposeful, but repeated losses at transitions, in procedural instructions, waiting and attention-management accumulate noticeably across the lesson. | ~75–90% is learning time. Explanation, questioning, discussion, practice, checking, feedback and closure fill most of the period; necessary transitions and routines are brief; students are rarely left without a purpose. |

Note the denominator: **scheduled** time, from the timetable — not the length of
the recording. A lesson that ends ten minutes early is penalised for those ten
minutes.

| ID | Question the system must answer | Sensor | Status |
| --- | --- | --- | --- |
| **A3-1** | **Second by second, what was the class actually doing?**<br>Label every second as one of: learning, settling, transition, behaviour, or dead/tech time. This one requirement effectively is the parameter. | Camera + Mic | Not built |
| **A3-2** | **What share of the scheduled period was learning time?**<br>Measured against the timetable, not against the recording. | Provided + Camera + Mic | Not built |
| **A3-3** | **How much time went on technology and setup problems?**<br>The teacher at the computer or projector, with nothing on screen the class can learn from. | Camera | **Partly built** — the screen is detected, but not whether it is showing anything useful |
| **A3-4** | **When the room was quiet, were students working — or waiting?** | Camera + Mic | Not built |
| **A3-5** | **How much of the talk had nothing to do with the lesson?**<br>The rubric's "unrelated talk". | Mic | Not built |
| **A3-6** | **Were there long stretches where students had no task at all?** | Camera | Not built — needs to see students |
| **A3-7** | **Which teaching activities actually happened?**<br>The Competent band names seven: explanation, questioning, discussion, practice, checking, feedback and closure. | Camera + Mic | Not built |
| **A3-8** | **How much time did students spend waiting while instructions were being given?**<br>Named in the Developing band as a recurring source of loss. | Camera + Mic | Not built |

### Why A3-4 is the clearest case in the whole domain

Thirty seconds of near-silence. Two possibilities:

- Heads down, pens moving. This is the most productive half-minute in the
  lesson.
- The teacher is fighting with the projector while the class stares into space.
  This is dead time.

A microphone scores these two identically — both are quiet. A camera alone
struggles as well, because in both cases students are sitting at their desks;
independent work and waiting-for-the-projector have the same posture. Putting
the two together resolves it: quiet plus pens moving is learning, quiet plus
nothing happening is not.

The entire learning-time percentage rests on getting this one distinction
right.

---

# Rules that apply to all three parameters

These come from the rubric's own closing note, quoted in full:

> - The system does not judge from a single movement, wherever the observation
>   is insufficient it states 'Not Observed'.
> - Lesson start/end is anchored to the time in the timetable.
> - Noise and movement are always read against the lesson's current phase —
>   silent independent work should be quiet; planned group work is expected to
>   be loud (this needs the lesson plan to be available or read the content
>   projected on the board or instruction given by the teacher).
> - Interpret with care where norms differ — labs, PE, library/practical work
>   (movement and noise are different), and lessons with no audible bell or
>   double periods, wherever the observation is unreliable it states that.

| ID | Question the system must answer | Sensor | Status |
| --- | --- | --- | --- |
| **X-1** | **What time was this lesson supposed to start and finish?**<br>Everything in A1 and A3 is judged against this. Without it, "late" and "share of the period" have no meaning at all. | Provided | Not built |
| **X-2** | **What was this part of the lesson meant to be?**<br>From the lesson plan if there is one; otherwise from what is projected on the board, or from what the teacher told the class to do. | Camera + Mic | Not built |
| **X-3** | **Is this noise level normal for what the class is doing right now?**<br>Silent independent work should be quiet. Group work is supposed to be loud. | Camera + Mic | Not built — depends on X-2 |
| **X-4** | **Is there enough evidence here, or is this one stray movement?**<br>The rubric forbids judging from a single movement. | Camera | **Partly built** — the pipeline already bridges gaps and requires signals to persist |
| **X-5** | **When the system cannot tell, does it say so?**<br>It must output "Not Observed" rather than guessing a band. | Provided | **Partly built** — quality tiers, a coverage score and an "unknown" state already exist |
| **X-6** | **Is this the kind of lesson where the normal rules don't apply?**<br>Labs, PE, library and practical work have different movement and noise norms; some rooms have no audible bell; double periods break the timing assumptions. In these cases the system must declare its reading unreliable. | Provided + Camera + Mic | Not built |
| **X-7** | **Can the system tell the teacher's voice from everyone else's?**<br>A prerequisite for most of A2. Either separating speakers automatically, or a clip-on microphone for the teacher plus a second one for room noise. | Mic | Not built |
| **X-8** | **Can the system see students at all?**<br>A prerequisite for six camera requirements. The detector needs to learn a student class. | Camera | Not built |

**X-3 deserves reading twice.** It means no fixed noise threshold is ever
valid. The same volume is Competent during group work and Emerging during
silent practice. Every audio measurement in A2 has to be interpreted against
what the class was *supposed* to be doing — which is why X-2 sits underneath so
much of this document.

---

# Glossary

| Term | What it means here |
| --- | --- |
| **Band** | One of the three grades: Emerging, Developing, Competent. |
| **T: / S:** | The rubric's shorthand. `T:` is something the teacher does, `S:` is something students do. |
| **First learning activity** | The moment real work begins — a starter, hook, recall exercise or discussion. Not the moment the teacher arrives. A1 is graded from here. |
| **Starter / do-now** | A short task waiting for students the moment they sit down, so the lesson begins without a gap. |
| **Plenary / closure** | The planned ending: a review, a reflection, an exit question, a summary. |
| **Displaced** | The rubric's term for a closure that technically happened but was drowned out by packing up. It does not count. |
| **Transition** | Any change from one activity to the next. |
| **Cue** | The signal a teacher uses to get attention — a phrase, a countdown, a hand. A familiar cue that works first time is a Competent trait. |
| **Settle time** | Seconds between the cue and the class actually being ready again. |
| **Redirection** | Correcting off-task behaviour. Competent is brief and early; Emerging is late and drawn-out. |
| **Off-task noise** | Noise that does not belong to the current activity — judged against the phase, never against a fixed volume. |
| **Dead time / tech time** | Time lost to equipment and setup that better preparation would have avoided. |
| **Learning-time share** | The percentage of the scheduled period spent learning. A3's score. |
| **Phase** | What the lesson is supposed to be doing right now — explanation, independent practice, group work. Sets the expectation for noise and movement. |
| **Not Observed** | The required output when there is not enough evidence to judge. Better than a wrong band. |
| **Speaker separation** | Working out who is talking. Needed because "raised voice" means the *teacher's* voice specifically. |
| **Detection class** | One of the kinds of object the model has been trained to recognise. Today: door, screen, teacher, pointing, writing. |
| **Zone** | A region marked on the video, such as the board or the doorway, used to tell where the teacher is. |
| **Heatmap** | A picture of where in the room the teacher spent their time. |
| **Coverage** | How much of the lesson the system could see the teacher for. Low coverage should trigger "Not Observed". |

---

# Open questions on the rubric

These need answering before any of this can be coded.

1. **A3 has no band above about 90%.** Competent tops out at "~75–90% is
   learning time". A lesson measured at 95% has nowhere to go. Is that a
   deliberate ceiling, or is a fourth band missing?
2. **There is no Exemplary band anywhere.** A1 and A2 also stop at Competent.
   Confirm the scale is meant to be three wide.
3. **Exactly 10 minutes late falls into two bands at once.** Emerging says
   "approximately 10 min late" and Developing says "approximately 5–10 min
   late". A tie-break rule is needed before this can be turned into code.
4. **Whose raised voice counts?** A2 grades raised voice, and the rubric's
   wording points at the teacher. Should a student shouting ever count against
   the teacher's A2 score, or only the teacher's own voice?
