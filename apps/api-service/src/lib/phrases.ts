/**
 * What a teacher's sentence is doing, read from its words.
 *
 * This is the stand-in for the labelling pass (docs/teacher-measurements.md,
 * Groups B-D): until an LLM labels utterances, these pattern tables do, and
 * every number built on them is reported as PROVISIONAL with the matched
 * sentence shown as evidence, so a reader can judge the match rather than
 * trust it.
 *
 * Two facts about the real transcripts shape the tables. Hindi comes back in
 * Devanagari, and so does a good deal of ENGLISH — "homework" arrives as
 * "होमवर्क", "tomorrow" as "टुमारो" — so every English cue has its Devanagari
 * spellings beside it. And dictation quotes the world: "the boys' bags were
 * kept outside" is a sentence being read out, not a pack-up instruction, so
 * the cues that end a lesson require an imperative, not a noun.
 */

export type ClosureType = "review" | "reflection" | "exit_question" | "summary";

export interface SentenceLabels {
  /** Launches work: "take out", "open page", "worksheet 18", "let us start". */
  setsTask: boolean;
  /** Calls for the class's attention: "listen", "quiet", "सुनो". */
  attentionCue: boolean;
  homework: boolean;
  /** Says the topic carries on next time. */
  continuation: boolean;
  /** Tells the class to pack up, close books, line up. */
  packUp: boolean;
  closure: ClosureType | null;
  /** Administrative talk: notebooks, planners, signatures, fees, attendance. */
  procedure: boolean;
}

const TASK =
  /\b(take out|open (your|the|to)|page (number|no\.?)?\s*\d|worksheet|work sheet|let'?s (start|begin|do|read|solve)|let us (start|begin|do|read|solve)|now (start|begin|write|read|solve)|just start|start (with|from|the|now|and|it)|do (question|the first|exercise|number)|question (number|no\.?)|exercise|write (down|the|this|it)|copy (this|the|down|it)|read (the|this|it|out|aloud)|solve|fill in|underline|circle|match the|answer (the|this|these)|discuss|complete (the|this|it|worksheet|exercise)|finish (it|this|the)|put a tick|tick|dictation)\b/iu;
const TASK_HI =
  /(टेक आउट|ओपन|पेज|वर्कशीट|वर्क शीट|स्टार्ट|शुरू (कर|हो|करते)|लिखो|लिख लो|लिखिए|राइट (डाउन|द|दिस|इट)|पढ़ो|पढ़ लो|पढ़िए|रीड (द|दिस|इट)|सॉल्व|कॉपी (कर|डाउन)|क्वेश्चन (नंबर|नं)|एक्सरसाइज|अंडरलाइन|टिक (कर|लगा)|चेक (कर|इट)|कंप्लीट (द|दिस|इट|कर)|फिनिश (इट|दिस|कर)|डिक्टेशन|मैप द)/u;

const ATTENTION =
  /\b(listen(ing|ed)?|quiet(en)?( up| down)?|silence|silent|attention|stop talking|no talking|not talk|don'?t talk|do not talk|look (here|at me|at the board)|eyes (here|on me|on the board)|settle down|sit down|sit properly|keep quiet|shut up|hello[!?]|excuse me)\b/iu;
const ATTENTION_HI =
  /(सुनो|सुनिए|सुन लो|ध्यान (दो|दीजिए|से|इधर)|चुप|शांत|साइलेंस|साइलेंट|लिसन|अटेंशन|बात मत|बातें मत|मत बोलो|इधर देखो|यहाँ देखो|बैठ जाओ|बैठो|सीधे बैठो|क्वाइट|क्वायट)/u;

const HOMEWORK =
  /\b(homework|home work|h\.?w\.?|at home|do (it|this|these) at home|for tomorrow|by tomorrow|bring (it|them|this|your \w+) tomorrow)\b/iu;
const HOMEWORK_HI =
  /(होमवर्क|होम वर्क|घर (पर|से|पे) (कर|पूरा|कंप्लीट|फिनिश)|घर से कर|कल (तक|लाना|लेकर|ले आना|ले के आना|करके लाना)|(ब्रिंग|कंप्लीट|फिनिश|डू) (इट|दिस|दैट|योर \S+) (टुमारो|टुमरो|टुमॉरो)|(टुमारो|टुमरो|टुमॉरो) (आईल|आई विल|यू आर|वन|यू विल) .*(होमवर्क|गिव|गेट))/u;

const CONTINUATION =
  /\b(next (class|period|lesson|time|week|day)|continue (this|it|from|with|tomorrow|next)|we('ll| will| shall) (continue|finish (this|it)|do (this|the rest|it)|take (this|it) up)|carry on (with this|next)|rest (of (this|it) )?(next|tomorrow|later)|remaining (part|questions?|worksheets?) (next|tomorrow|later)|to be continued)\b/iu;
const CONTINUATION_HI =
  /(नेक्स्ट (क्लास|पीरियड|टाइम|डे|वीक)|कंटिन्यू|अगले (पीरियड|क्लास|दिन|हफ़्ते|हफ्ते)|अगली (क्लास|बार)|कल (करेंगे|पढ़ेंगे|देखेंगे|कंटिन्यू|पूरा करेंगे|फिनिश करेंगे)|बाकी (कल|अगले|अगली|बाद में)|जारी रखेंगे|आगे (कल|अगली))/u;

const PACK_UP =
  /\b(pack (up|your)|packup|bags? up|put (your \w+ )?(away|back|inside)|close your (books?|copies|notebooks?|worksheets?)|keep your (books?|copies|notebooks?|things) (away|inside|back|in (your|the) bags?)|line up|wind(ing)? up|time'?s up|time is up|time over|that'?s (all|it) for today|class (is )?over|period (is )?over)\b/iu;
const PACK_UP_HI =
  /(पैक (अप|कर)|पैकअप|बैग (में|उठा|रख|पैक|बंद)|बैग्स (में|उठा|रख|पैक)|किताब(ें|े)? (बंद|रख|अंदर)|कॉपी (बंद|रख)|कॉपियाँ (बंद|रख)|बुक्स (क्लोज़|क्लोज|बंद|कीप|रख)|(क्लोज़|क्लोज|कीप) योर (बुक्स|बुक|किताब|किताबें|कॉपी|कॉपियां|कॉपीज|बैग|बैग्स|थिंग्स)|लाइन (अप|में|बना)|टाइम (अप|ओवर|खत्म)|समय (खत्म|समाप्त|हो गया)|पीरियड (ओवर|खत्म)|क्लास (ओवर|खत्म)|चलो (उठो|निकलो))/u;

const CLOSURE: [ClosureType, RegExp][] = [
  [
    "review",
    /\b(revise|revision|recap|let'?s (revise|recap|review|go over|go through)|let us (revise|recap|review)|what (did|have) we (learn|learnt|learned|do|cover)(ed)?|what we (learnt|learned|did|covered) today|quick(ly)? (revise|recap|review)|रिवाइज|रिवीजन|रीकैप|रिव्यू|दोहरा|आज हमने (क्या )?(सीखा|पढ़ा|किया))\b/iu,
  ],
  [
    "summary",
    /\b(summar(y|ise|ize|izing|ising)|to sum up|in summary|in short|conclusion|to conclude|so today we (have )?(learnt|learned|covered|did|studied)|समरी|संक्षेप|निष्कर्ष|आज हमने .*(सीखा|पढ़ा|किया))\b/iu,
  ],
  [
    "reflection",
    /\b(how did you (feel|find)|what was (difficult|easy|hard|new)|reflect|think about what (you|we)|what did you (like|find)|कैसा लगा|क्या मुश्किल|क्या नया)\b/iu,
  ],
  [
    "exit_question",
    /\b(before (you|we) (go|leave)|one (last|final|quick) question|exit (ticket|question|slip)|last question|quick question before|जाने से पहले|आखिरी सवाल|एक (आखिरी|लास्ट) (सवाल|क्वेश्चन)|लास्ट क्वेश्चन)\b/iu,
  ],
];

const PROCEDURE =
  /\b(notebooks?|copies|planner|almanac|diary|sign(ature|ed)?|submit|collect(ing)?|attendance|roll (number|no\.?)|fees?|monitor|stapler|register|uniform|bus|lunch|tiffin|water bottle|absent|present (sir|ma'?am)|circular|permission|principal)\b/iu;
const PROCEDURE_HI =
  /(नोटबुक|नोट बुक|कॉपियां|प्लानर|अलमनैक|डायरी|साइन|सिग्नेचर|सबमिट|कलेक्ट|अटेंडेंस|रोल नंबर|रोल नं|फीस|मॉनिटर|स्टेपलर|रजिस्टर|यूनिफॉर्म|यूनिफार्म|बस|लंच|टिफिन|खाना|पानी की बोतल|बोतल|प्रिंसिपल|सर्कुलर|परमिशन)/u;

function closureOf(text: string): ClosureType | null {
  for (const [type, re] of CLOSURE) if (re.test(text)) return type;
  return null;
}

export function labelSentence(text: string): SentenceLabels {
  const t = text.trim();
  return {
    setsTask: TASK.test(t) || TASK_HI.test(t),
    attentionCue: ATTENTION.test(t) || ATTENTION_HI.test(t),
    homework: HOMEWORK.test(t) || HOMEWORK_HI.test(t),
    continuation: CONTINUATION.test(t) || CONTINUATION_HI.test(t),
    packUp: PACK_UP.test(t) || PACK_UP_HI.test(t),
    closure: closureOf(t),
    procedure: PROCEDURE.test(t) || PROCEDURE_HI.test(t),
  };
}
