<!-----



Conversion time: 1.995 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs™ to Markdown version 2.0β1
* Wed Dec 31 2025 04:57:53 GMT-0800 (PST)
* Source doc: Creating a Beginner's RLHF Book
* Tables are currently converted to HTML tables.
----->



# Foundations of Reinforcement Learning from Human Feedback and Model Post-Training


## Introduction to the Intelligence Cake and the Racing Chassis

The development of modern artificial intelligence is often described through the "Intelligence Cake" metaphor, a conceptual framework popularized by researchers to explain the relative importance of different training phases.<sup>1</sup> In this visualization, the bulk of the cake represents self-supervised pre-training, where a model consumes trillions of words from the public internet to learn the fundamental structures of grammar, logic, and factual associations.<sup>2</sup> The icing on the cake is supervised fine-tuning (SFT), which prepares the model for specific formats like question-answering.<sup>4</sup> Finally, the reinforcement learning from human feedback (RLHF) phase is the cherry on top—a small but critical layer that aligns the model’s vast potential with the specific, often subtle preferences of human users.<sup>1</sup>

To understand why this "cherry" is so impactful, one may consider the "Car Chassis" analogy.<sup>1</sup> A base model emerging from pre-training is like the engine and chassis of a high-performance racing vehicle. It possesses immense power and mechanical potential, but it is effectively undriveable for the average consumer. Post-training is the process by which engineers add the steering wheel, the seats, the specialized aerodynamics, and the fine-tuned suspension.<sup>1</sup> Just as a Formula 1 team can significantly improve a car's lap times throughout a season without changing the core engine, the post-training team extracts latent performance and utility from a static base model.<sup>1</sup>

RLHF specifically addresses the "Elicitation Theory of Post-training," which suggests that much of the intelligence we see in models like ChatGPT was already present in the base model but remained inaccessible.<sup>1</sup> RLHF provides the interface that allows users to "elicit" this knowledge reliably. It transitions the model from being a simple text-completion engine into a helpful assistant.<sup>1</sup> This transition is not merely about learning new facts; it is about learning a style, a tone, and a set of behavioral boundaries that make the AI a safer and more enjoyable partner for human interaction.<sup>1</sup>


### Revision: The Basics of AI Developmental Stages

Modern AI development occurs in layers. Pre-training creates a broad foundation of knowledge. Supervised fine-tuning teaches the model to follow a specific conversational format. Reinforcement Learning from Human Feedback acts as the final refinement, ensuring the model behaves according to human values and preferences. This entire suite of techniques after the initial pre-training is collectively referred to as post-training.


### Comprehension Questions



1. How does the "Intelligence Cake" metaphor explain the distribution of training effort in AI?
2. Using the "Car Chassis" analogy, what corresponds to the base model and what corresponds to post-training?
3. What is the core goal of the "Elicitation Theory of Post-training"?
4. Why is the base model alone considered insufficient for most consumer applications?


## Historical Evolution and Seminal Milestones

The formalization of RLHF did not happen in isolation but emerged from the convergence of economics, psychology, and optimal control theory.<sup>1</sup> Early research into learning from preferences was conducted in the context of robotics and simple games. One of the foundational frameworks was TAMER (Training an Agent Manually via Evaluative Reinforcement), where human trainers provided real-time feedback to guide an agent's learning process.<sup>1</sup> This was later expanded upon by researchers who demonstrated that agents could learn complex behaviors, such as performing a backflip in a simulated environment, using only 900 bits of human feedback.<sup>1</sup>

The shift toward language modeling began in earnest between 2019 and 2022. During this period, techniques were developed to fine-tune GPT-2 and early GPT-3 models on human preferences for tasks like summarization and instruction-following.<sup>1</sup> The announcement of ChatGPT at the end of 2022 marked the "ChatGPT Era," where the industry at large recognized RLHF as a prerequisite for creating viable consumer products.<sup>1</sup>


<table>
  <tr>
   <td><strong>Milestone</strong>
   </td>
   <td><strong>Year</strong>
   </td>
   <td><strong>Significance</strong>
   </td>
  </tr>
  <tr>
   <td>TAMER Framework
   </td>
   <td>2008
   </td>
   <td>Introduced manual evaluative reinforcement for agent training.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td>Christiano et al.
   </td>
   <td>2017
   </td>
   <td>Applied RLHF to deep reinforcement learning in Atari environments.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td>Summarization Study
   </td>
   <td>2020
   </td>
   <td>Demonstrated RLHF's efficacy in generating high-quality text summaries.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td>InstructGPT
   </td>
   <td>2022
   </td>
   <td>Established the canonical three-stage RLHF pipeline for language models.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td>Constitutional AI
   </td>
   <td>2022
   </td>
   <td>Introduced AI feedback as a scalable alternative to human feedback.<sup>1</sup>
   </td>
  </tr>
</table>


The current state of the field is characterized by a transition from standard RLHF toward more specialized techniques, such as Reinforcement Learning with Verifiable Rewards (RLVR) and Direct Alignment Algorithms (DAAs) like DPO.<sup>1</sup> This evolution reflects a growing need for models that can reason through complex math and code while maintaining a consistent and helpful personality.<sup>1</sup>


### Revision: The Timeline of RLHF

The journey of RLHF began with simple control tasks in robotics before being adapted for complex language tasks. Key models like InstructGPT laid the groundwork for the modern AI assistants we use today. The field is now moving toward methods that use less human data and more automated "verifiable" checks to improve reasoning.


### Comprehension Questions



1. What was the primary contribution of the TAMER framework to RLHF history?
2. Which specific text-based task first proved that RLHF could outperform supervised learning in 2020?
3. How did the InstructGPT model change the industry's approach to alignment?
4. What is the fundamental difference between the early RLHF work in 2017 and the work done on language models in 2022?


## Mathematical Foundations and Core Definitions

To navigate the technical landscape of RLHF, one must establish a rigorous vocabulary rooted in both natural language processing (NLP) and reinforcement learning (RL).<sup>1</sup> A "Prompt" is defined as the input text ($x$) given to a model, while the "Completion" ($y$) is the generated response.<sup>1</sup> Within a pairwise setting, we distinguish between the "Chosen Completion" ($y_c$), which represents the preferred answer, and the "Rejected Completion" ($y_r$), which represents the disfavored alternative.<sup>1</sup>

The "Policy" ($\pi$) is a crucial concept, representing the probability distribution over all possible completions given a prompt.<sup>1</sup> During training, we distinguish between the "Reference Model" ($\pi_{ref}$), which is the static starting point, and the "Policy Model" ($\pi_{\theta}$), which is the active model being updated.<sup>1</sup> The goal of the RLHF process is to optimize this policy model to maximize a "Reward" ($r$), a scalar value indicating the desirability of a response.<sup>1</sup>

Modern language modeling relies on "Autoregression," a mechanism where each next word (or token) predicted by the model depends on the sequence of tokens that came before it.<sup>1</sup> Mathematically, for a sequence $x = (x_1, x_2, \dots, x_T)$, the joint probability is factorized as:

$$P_{\theta}(x) = \prod_{t=1}^{T} P_{\theta}(x_t | x_1, \dots, x_{t-1})$$

In this framework, the model is trained by minimizing the "Negative Log-Likelihood" (NLL) of the training data, effectively teaching the model to make the correct next token as probable as possible.<sup>1</sup>


### Revision: Defining the Landscape

An AI's behavior is dictated by its "Policy," which is just a fancy word for its decision-making rules. When we train the AI, we use a "Reward" to tell it if its "Completion" was good. Most of this math happens at the "Token" level, which are the small pieces of words that the AI processes one by one in a process called "Autoregression."


### Comprehension Questions



1. Define the variables $x$ and $y$ in the context of a language model prompt.
2. What is the difference between a reference model and a policy model?
3. Explain the term "Autoregression" in simple words.
4. Why is the "Chosen Completion" denoted as $y_c$?


## The RLHF Training Overview and Problem Formulation

The optimization of RLHF is a multifaceted engineering challenge that adapts the standard reinforcement learning loop for the unique constraints of language.<sup>1</sup> In traditional RL, an agent interacts with an "Environment," takes "Actions," and receives "Rewards" based on state transitions.<sup>13</sup> A common example is the "CartPole" task, where an agent must move a cart left or right to keep a pole balanced.<sup>1</sup> In that scenario, the reward is a simple $+1$ for every second the pole remains upright.<sup>1</sup>

Language models, however, operate differently. The "Environment" is effectively the prompt dataset, and the "Action" is the generation of an entire sequence of tokens.<sup>1</sup> Unlike a robot moving through a physical space, the language model’s action (its response) typically does not change the state of the next prompt.<sup>1</sup> This makes RLHF more similar to a "Contextual Bandit" problem, where we care about the reward of a single, complete response rather than a long sequence of interconnected moves.<sup>1</sup>

The core RLHF objective involves maximizing the expected reward while staying close to the original model to prevent the behavior from becoming nonsensical.<sup>1</sup> This is expressed as:

$$J(\pi) = \mathbb{E}*{\tau \sim \pi} [r*{\theta}(s_t, a_t)] - \beta D_{KL}(\pi_{RL} |

| \pi_{ref})$$

In this equation, the first term seeks to maximize the points given by the reward model, while the second term (the KL Divergence) acts as a penalty if the model drifts too far from its original instruction-following behavior.<sup>1</sup>


<table>
  <tr>
   <td><strong>Aspect</strong>
   </td>
   <td><strong>Standard RL</strong>
   </td>
   <td><strong>RLHF for Language Models</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Reward Signal</strong>
   </td>
   <td>Environmental Function
   </td>
   <td>Learned Human Preference Model.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>State Transitions</strong>
   </td>
   <td>Dynamic Environment
   </td>
   <td>Static Prompts from a Dataset.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>Action</strong>
   </td>
   <td>Physical Move
   </td>
   <td>Token Sequence Generation.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>Reward Granularity</strong>
   </td>
   <td>Per-step / Fine-grained
   </td>
   <td>Usually Response-level (Bandit-style).<sup>1</sup>
   </td>
  </tr>
</table>



### Revision: The Rules of the Game

RLHF treats writing a response like a game where the AI earns points for being helpful. Unlike a robot in a physical world, the AI is playing a "one-turn" game for every question. To keep the game fair and prevent the AI from "cheating" or acting crazy, we subtract points if it stops sounding like its original, human-taught self.


### Comprehension Questions



1. How does the "CartPole" example help explain the concept of a reward?
2. Why is RLHF often compared to a "Bandit" problem instead of a traditional multi-step RL problem?
3. What is the purpose of the $\beta$ (beta) value in the RLHF objective?
4. In what way is the "Reward" in RLHF different from a "Reward" in a video game?


## The Nature and Philosophy of Human Preferences

To build an AI that understands human values, researchers must first confront the complexity of what a "preference" actually is.<sup>1</sup> The term traces its origins to early philosophical discussions in Aristotle’s *Topics* and developed through Bentham’s utilitarian "Hedonic Calculus"—the belief that all pleasure and pain could be weighed on a single scale.<sup>1</sup> In modern RLHF, the "VNM Utility Theorem" provides the theoretical license to represent these messy human values as a single numerical "Utility" function.<sup>1</sup>

However, human preferences are notoriously unstable and contextual. A behavioral economist might argue that preferences do not strictly exist but are instead "revealed" through choices.<sup>1</sup> For example, if a human is asked to choose between two poems about an optimistic goldfish, their choice might depend on their current mood, their cultural background, or even the order in which the poems were presented.<sup>1</sup> This phenomenon is known as "Position Bias" or "Preference Shift".<sup>1</sup>

The challenge for RLHF designers is "Aggregation"—the process of combining the conflicting values of thousands of different human labelers into one single model.<sup>1</sup> "Impossibility Theorems" in social choice theory suggest that it is mathematically impossible to satisfy everyone’s fairness criteria simultaneously.<sup>1</sup> Therefore, modern models are often optimized for "Empirical Alignment"—maximizing average performance across a population rather than adhering to a single, perfect moral code.<sup>1</sup>


### Revision: Understanding Human Values

Preferences aren't hard rules; they are comparative choices. Because every human is different, the AI has to learn an "average" of what thousands of people find helpful. This is hard because humans aren't always consistent—sometimes we pick things just because they appear first or sound more confident, even if they aren't better.


### Comprehension Questions



1. How does "Hedonic Calculus" relate to the math of AI alignment?
2. What does the "VNM Utility Theorem" allow AI researchers to do?
3. What is "Revealed Preference," and how does it apply to labeling AI data?
4. Why is "Aggregation" a difficult mathematical problem in social choice theory?


## Preference Data Collection and Sourcing

Preference data is the fuel that powers the RLHF engine. Collecting this data is one of the most opaque and expensive parts of the entire AI pipeline, often involving millions of dollars in contracts with data vendors like Scale AI.<sup>1</sup> The process begins with "Prompt Engineering," where humans write a diverse set of questions for the model to answer.<sup>3</sup> The model then generates two or more completions for each prompt, and humans are asked to rank them.<sup>1</sup>

Interface design plays a critical role in the quality of this data. Most labels are collected using a "Likert Scale," where a rater indicates the strength of their preference between two options (e.g., "A is much better than B"). A common alternative is a simple "Binary Ranking," where the rater merely clicks on the better answer.<sup>20</sup>


<table>
  <tr>
   <td><strong>Ranking Method</strong>
   </td>
   <td><strong>Structure</strong>
   </td>
   <td><strong>Advantage</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Ratings (1-5)</strong>
   </td>
   <td>Score each response individually.
   </td>
   <td>Provides metadata on absolute quality.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>Rankings (A vs B)</strong>
   </td>
   <td>Compare two responses and pick the winner.
   </td>
   <td>More consistent and less noisy for humans.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>Multi-turn</strong>
   </td>
   <td>Preferences over a whole conversation.
   </td>
   <td>Essential for training chatbot consistency.<sup>1</sup>
   </td>
  </tr>
</table>


Operational challenges are rampant in this stage. "Vendor Sourcing" requires building relationships with professional workforces of labelers.<sup>1</sup> These labelers must be given "Instruction Manuals" hundreds of pages long to ensure they are rating responses based on the lab’s specific goals, such as prioritizing "Truthfulness" over "Politeness".<sup>1</sup> If the instructions are unclear, the model may inherit the labelers’ biases, such as "Verbosity Bias"—a tendency to prefer longer answers regardless of their accuracy.<sup>1</sup>


### Revision: How We Get Human Data

To get human feedback, we show a person two different AI answers and ask them to pick the winner. This is usually done by professional workers who follow strict rulebooks. We prefer "Contests" (A vs B) over "Grades" (1 to 10) because people are much better at comparing things than assigning a perfect score to a single sentence.


### Comprehension Questions



1. Why is human preference data considered a "moat" or competitive advantage for big AI companies?
2. Explain the difference between a ranking and a rating.
3. What is "Verbosity Bias" in the context of data labeling?
4. Why are human instruction manuals for labelers often hundreds of pages long?


## Reward Modeling: The Bradley-Terry Framework

The Reward Model (RM) is the mathematical heart of RLHF, acting as a digital stand-in for human judgment.<sup>1</sup> Its goal is to take any piece of text and output a scalar "Reward Logit" that predicts how much a human would like it.<sup>1</sup> This is achieved using the "Bradley-Terry Model" of preferences, which assumes that the probability of choosing answer $y_c$ over $y_r$ is a function of the difference in their underlying scores <sup>1</sup>:

$$P(y_c \succ y_r | x) = \sigma(r_{\theta}(x, y_c) - r_{\theta}(x, y_r)) = \frac{1}{1 + e^{-(r_{\theta}(y_c) - r_{\theta}(y_r))}}$$

To train this model, we minimize a "Contrastive Loss" function. If the Reward Model gives a low score to the answer the human liked, the loss is high, and the model updates its weights to correct the mistake.<sup>1</sup> This effectively compresses the complex, qualitative world of human values into a single, high-dimensional score space.<sup>1</sup>

Recent innovations have introduced "Process Reward Models" (PRMs), which are particularly effective for reasoning tasks.<sup>1</sup> While a standard RM only looks at the final answer, a PRM evaluates every step of a model’s logic. This is analogous to a math teacher who gives partial credit for the "work shown" rather than just looking at the final number on the page.<sup>23</sup> This "Fine-grained" feedback allows models to learn *how* to reason, not just *what* the answer should be.


### Revision: Building the Digital Referee

A Reward Model is like a "Point Scorer" for text. We use the Bradley-Terry math formula to turn human votes into numbers. Some Reward Models grade the whole essay at once, while "Process" models grade every single sentence, giving the AI "Partial Credit" for its thinking.


### Comprehension Questions



1. Write the Bradley-Terry probability formula and explain what the variable $r$ represents.
2. What is a "Contrastive Loss," and how does it help train a referee?
3. How does a Process Reward Model (PRM) differ from a standard Reward Model?
4. In the math teacher analogy, why is "Partial Credit" better for training logic?


## Regularization: The Invisible Anchor and Rubber Band

A significant risk in RLHF is "Over-optimization," also known as "Reward Hacking".<sup>1</sup> Because the Reward Model is an imperfect proxy for a human, the model being trained will eventually find "exploits" to get a high score.<sup>24</sup> For example, the model might learn that including certain "magic" phrases or formatting everything in bullet points tricks the Reward Model into giving it 100 points, even if the answer is wrong.

To prevent this, researchers use "Regularization," with the "Kullback-Leibler (KL) Divergence" being the most popular tool.<sup>1</sup> One can think of KL Divergence as a "Rubber Band" connecting the new model back to the original, human-taught SFT model.<sup>8</sup> If the new model tries to change its behavior too drastically, the rubber band stretches, and the "KL Penalty" increases, lowering the total reward.<sup>1</sup>

Mathematically, the KL Divergence measures the "Expected Surprisal" of using the new model to approximate the old one.<sup>27</sup> If the new model puts high probability on a word that the old model thought was impossible, the "Surprise" is high, and the penalty is severe.<sup>27</sup> This ensures that the AI stays "grounded" and doesn't drift into alien or broken speech patterns during its pursuit of high scores.<sup>1</sup>


### Revision: The Penalty for Acting Weird

When the AI is practicing, it might figure out "cheat codes" to get high scores. We use KL Divergence to measure how much the AI has changed. If it starts talking in a weird way that the original model wouldn't have used, we take points away. It’s like an anchor that keeps a ship from floating too far away from the harbor during a storm.


### Comprehension Questions



1. Why is KL Divergence called a "Distance Measure," and why is that term technically a bit inaccurate?
2. What is the "Surprise" analogy for KL math?
3. How does "Reward Hacking" relate to "Goodhart’s Law"?
4. If a model starts talking in a totally different language to get a high score, how does the KL penalty stop it?


## Supervised Fine-Tuning (SFT) and Instruction Tuning

Supervised Fine-Tuning (SFT), often called "Instruction Tuning," is the prerequisite phase that sets the stage for RLHF.<sup>1</sup> A base model is essentially a high-end "Autocomplete" machine; if asked "What is the capital of France?", it might respond with "What is the capital of Germany?" because it thinks it is reading a list of trivia questions.<sup>5</sup> SFT changes this by showing the model thousands of $(Prompt, Response)$ pairs written by experts.<sup>1</sup>

The implementation of SFT relies on "Chat Templates," which are snippets of code (usually in a language called Jinja) that wrap user messages in special boundary tokens.<sup>1</sup> This creates a "Language of the Assistant" that the model learns to speak fluently.<sup>1</sup>


<table>
  <tr>
   <td><strong>Term</strong>
   </td>
   <td><strong>Role in SFT</strong>
   </td>
   <td><strong>Analogy</strong>
   </td>
  </tr>
  <tr>
   <td><strong>System Prompt</strong>
   </td>
   <td>High-level behavioral instructions.
   </td>
   <td>The character's "Backstory." <sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>User Message</strong>
   </td>
   <td>The specific question or request.
   </td>
   <td>The "Script" from the actor's partner. <sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>Assistant Role</strong>
   </td>
   <td>The model's generated response.
   </td>
   <td>The actor's "Performance." <sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>EOS Token</strong>
   </td>
   <td>Signals the model has finished talking.
   </td>
   <td>The "Curtain Call." <sup>1</sup>
   </td>
  </tr>
</table>


Modern best practices suggest that "Quality is more important than Quantity" in SFT.<sup>1</sup> While models can be trained on millions of samples, a few thousand extremely high-quality examples can often achieve 90% of the possible alignment.<sup>1</sup> This stage is responsible for the model's "Format Following"—its ability to respond in lists, markdown, or specific coding languages.<sup>1</sup>


### Revision: Teaching the Format

SFT is like giving the AI a job description and a few thousand examples of good work. It teaches the AI that when a human asks a question, it should provide an answer instead of just continuing the sentence. It also teaches the AI "Chat Markup," which is the secret code it uses to tell its own voice apart from the user's voice.


### Comprehension Questions



1. Why is a base model alone bad at answering questions?
2. What is a "Chat Template," and why does it use special tokens like &lt;|im_start|>?
3. In SFT, why do we "mask" the user's prompt during the training process?
4. Explain the difference between "Knowledge" (from pre-training) and "Format" (from SFT).


## Rejection Sampling: Selection and Iteration

Rejection Sampling is an intuitive "Shortcut" to preference alignment that does not require the complexity of reinforcement learning.<sup>1</sup> The core idea is that even a mediocre model will occasionally produce a "brilliant" answer by chance.<sup>30</sup> Rejection sampling works by generating many answers (usually between 10 and 100) for every question in the training set.<sup>1</sup>

Once the answers are generated, they are scored by a Reward Model.<sup>1</sup> We then "Reject" the bad answers and keep only the "Best" answer for each prompt.<sup>31</sup> Finally, we perform a standard round of SFT on these high-scoring samples.<sup>1</sup> This creates a "Data Flywheel": the model is essentially learning from the very best versions of itself.<sup>1</sup>

A useful non-technical analogy for this is the "Talent Show".<sup>30</sup> Imagine an AI that performs 50 different versions of a dance routine. A judge watches all 50 and throws away the 49 videos where the AI tripped or forgot a move. We then show the AI the one video of its "Perfect Performance" and tell it: "Always dance like this".<sup>30</sup> This method was a major contributor to the success of Llama 2 and continues to be used in modern "Thinking" models to instill specific reasoning habits.<sup>1</sup>


### Revision: Learning from Your Best Days

Rejection Sampling is a way to filter the AI's practicing. The AI tries to answer the same question many times, and we only keep the answers that get "Gold Stars" from our Reward Model judge. We then use those "Gold Star" answers to give the AI a new set of lessons.


### Comprehension Questions



1. Walk through the four steps of a Rejection Sampling pipeline.
2. Why is Rejection Sampling considered "computationally expensive" during the generation phase?
3. What is the difference between "Best-of-N" at inference time and Rejection Sampling during training?
4. How does Rejection Sampling help "denoise" human data?


## Reinforcement Learning: Policy Gradients and PPO

Reinforcement Learning (RL) is the most complex stage of the pipeline, involving "Policy Gradient" algorithms that update the model’s weights based on trial and error.<sup>1</sup> The fundamental "Trick" of policy gradients is the "Log-Derivative Trick," which allows researchers to turn a "High Reward" into a "High Probability" for a specific sequence of tokens.<sup>12</sup>

Proximal Policy Optimization (PPO) is the industry standard for this task.<sup>1</sup> PPO addresses the problem of "Gradient Variance"—the fact that a single lucky guess or unlucky mistake can cause the model to over-correct its parameters.<sup>12</sup> To solve this, PPO uses "Importance Sampling" and a "Clipping Objective".<sup>1</sup>

The "Clipping" mechanism acts like a "Cautious Coach".<sup>33</sup> When the model finds a move that gets a high reward, the coach says, "That was good, but only increase the likelihood of that move by 20% today".<sup>34</sup> This prevents the model from "collapsing" into a state where it only knows how to say one thing. PPO also learns a "Value Function"—a separate internal model that tries to predict how many points a completion *will* get, which helps the model stay focused on long-term quality rather than just the next word.<sup>1</sup>


### Revision: The Science of Trial and Error

RL is how the AI "practices" its conversational skills. PPO is the most common algorithm for this. It uses a "Clip" to make sure the AI doesn't change its whole personality too fast after one good grade. It also uses a "Value Function" to act as a second brain that predicts the future score of a sentence.


### Comprehension Questions



1. Explain the "Log-Derivative Trick" in a non-technical way.
2. What does it mean for an algorithm to be "On-Policy"?
3. How does PPO clipping prevent a model from "exploding" during training?
4. What is the role of the "Value Function" in the PPO algorithm?


## Direct Preference Optimization (DPO) and Direct Alignment

In 2023, a breakthrough paper titled "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" introduced a simpler way to align models without needing RL at all.<sup>1</sup> DPO mathematically proves that for any RLHF objective, there is a "Closed-form Solution" that can be reached by simply comparing the probabilities of two completions.<sup>1</sup>

The DPO loss function looks at the "Log-Ratio" of probabilities. It asks: "How much more likely is the chosen answer under the new model compared to the old model?".<sup>1</sup>

$$L_{DPO}(\pi_{\theta}; \pi_{ref}) = -\mathbb{E}_{(x, y_c, y_r) \sim D} [ \log \sigma ( \beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{ref}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{ref}(y_r|x)} ) ]$$

DPO is like "Learning Chess by Studying a Chess Manual" rather than playing millions of games.<sup>12</sup> It is much easier to implement because it only requires two versions of the model (the one you are training and a reference copy) and does not need the complex "Value Network" or "Clipping" logic of PPO.<sup>1</sup> However, DPO can suffer from "Likelihood Displacement," where the model occasionally lowers the probability of *both* answers, making it less robust in unfamiliar situations.


<table>
  <tr>
   <td><strong>Algorithm</strong>
   </td>
   <td><strong>Method</strong>
   </td>
   <td><strong>Major Benefit</strong>
   </td>
  </tr>
  <tr>
   <td><strong>PPO</strong>
   </td>
   <td>Active practice (online).
   </td>
   <td>Can discover new, high-quality behaviors.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>DPO</strong>
   </td>
   <td>Classification (offline).
   </td>
   <td>Stable, fast, and uses less memory.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>SimPO</strong>
   </td>
   <td>Reference-free DPO.
   </td>
   <td>Simpler math; no reference model needed.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>KTO</strong>
   </td>
   <td>Prospect theory based.
   </td>
   <td>Works on single "thumbs-up" data points.<sup>1</sup>
   </td>
  </tr>
</table>



### Revision: The Simple Alternative

DPO is a newer, faster way to align AIs. Instead of letting the AI practice and play games, we just show it the human preferences and use math to "teleport" the AI to the right behavior. It is like a student who learns the answers to the test by heart instead of studying the whole subject.


### Comprehension Questions



1. Why is DPO often called "Offline" RL?
2. What is the "Closed-form Solution" that DPO relies on?
3. Explain "Likelihood Displacement" in DPO.
4. Why does DPO require less computer memory than PPO?


## AI Feedback, Constitutional AI, and RLAIF

As the scale of AI training increased, the demand for human data became a bottleneck.<sup>1</sup> Researchers at Anthropic introduced "Constitutional AI" (CAI) as a solution, allowing models to be aligned with "AI Feedback" instead of human feedback.<sup>1</sup> This process is known as Reinforcement Learning from AI Feedback (RLAIF).<sup>18</sup>

CAI works in two stages. First, in the "Supervised Stage," a model is given a "Constitution"—a set of written rules like "Choose the response that is least likely to be viewed as harmful".<sup>1</sup> The model generates a response, critiques it based on the constitution, and then rewrites it.<sup>1</sup> This creates a high-quality "Synthetic Dataset".<sup>38</sup> Second, in the "RL Stage," a "Judge Model" (usually a larger AI like GPT-4) acts as the referee, scoring the model’s answers based on those same rules.<sup>1</sup>

The "AI Bill of Rights" analogy is perfect here.<sup>39</sup> CAI is about giving the AI a "Soul" or a "Moral Compass" that is explicitly written in English rather than implicitly learned from noisy human rankings.<sup>36</sup> This makes the alignment process more "Transparent"—researchers can look at the constitution and know exactly why the AI refused to answer a specific question.<sup>1</sup>


### Revision: The AI Bill of Rights

Constitutional AI is a way to teach the AI to be "good" using a rulebook instead of human ratings. The AI uses its rules (the Constitution) to grade itself and fix its own mistakes. This is called RLAIF, and it is much faster and cheaper than paying thousands of humans to rate answers.


### Comprehension Questions



1. What are the two phases of Constitutional AI?
2. How does a "Constitution" differ from a "Reward Model"?
3. What is "Self-Critique" in the context of CAI?
4. Why is RLAIF considered more "scalable" than RLHF?


## Reasoning Models and RL with Verifiable Rewards

A new frontier emerged in late 2024 with "Thinking Models" like OpenAI’s o1 and DeepSeek R1.<sup>1</sup> These models are trained to perform "Chain of Thought" (CoT) reasoning—visibly or hidden—before providing an answer.<sup>23</sup> The training method used here is Reinforcement Learning with Verifiable Rewards (RLVR).<sup>1</sup>

In RLVR, we don't need a Reward Model to guess if a math answer is good.<sup>1</sup> Instead, we use a "Verification Function"—a simple piece of software that checks if the AI’s answer is correct (e.g., checking if the final number is 42 or if a computer program passes its tests).<sup>1</sup> This provides a "Ground Truth" signal that is 100% accurate.

A useful analogy for this behavior is "Rubber Duck Debugging".<sup>17</sup> Programmers often find that by explaining their code to an inanimate object, they catch their own errors.<sup>17</sup> Reasoning models do the same: they "talk to themselves" in a hidden scratchpad to work through difficult logic.<sup>42</sup> This allows for "Inference-time Scaling," where we can make an AI "smarter" simply by giving it more time (and more tokens) to think before it speaks.<sup>1</sup>


### Revision: Showing the Work

Reasoning models are like students who show their work on a math test. We train them using RLVR, where the computer gives them a point only if they get the right answer. Because the computer can check math perfectly, the AI can practice millions of times without any humans involved until it becomes an expert at logic.


### Comprehension Questions



1. What is "Chain of Thought" (CoT)?
2. How does a Verification Function improve training for math and code?
3. What is the "Rubber Duck Debugging" analogy in reasoning models?
4. Explain "Inference-time Scaling" in your own words.


## Tool Use, Function Calling, and the Model Context Protocol

As AI becomes more integrated into professional workflows, it must learn to use "Tools"—external software like calculators, web browsers, or corporate databases.<sup>1</sup> This is known as "Function Calling".<sup>1</sup> The AI is trained to output specific tokens that trigger a tool, wait for the result, and then incorporate that result into its final answer.<sup>1</sup>

The "Handyman’s Tool Belt" is the standard analogy for this skill.<sup>44</sup> A handyman (the AI) has knowledge in his head, but he can only fix a sink if he has a wrench (the tool).<sup>44</sup> The "Model Context Protocol" (MCP) is a new open standard that acts as a "Universal Connector" for these tools.<sup>47</sup>


<table>
  <tr>
   <td><strong>MCP Component</strong>
   </td>
   <td><strong>Role</strong>
   </td>
   <td><strong>Analogy</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Host</strong>
   </td>
   <td>Where the AI lives (e.g., the browser).
   </td>
   <td>The "House" being repaired. <sup>46</sup>
   </td>
  </tr>
  <tr>
   <td><strong>Client</strong>
   </td>
   <td>The interface that talks to tools.
   </td>
   <td>The "Waiter" taking orders. <sup>46</sup>
   </td>
  </tr>
  <tr>
   <td><strong>Server</strong>
   </td>
   <td>Where the tools and data live.
   </td>
   <td>The "Kitchen" preparing data. <sup>46</sup>
   </td>
  </tr>
  <tr>
   <td><strong>Tools</strong>
   </td>
   <td>Executable functions (APIs).
   </td>
   <td>The "Recipes" or "Tools" themselves. <sup>46</sup>
   </td>
  </tr>
</table>


Training an AI for tool use is a delicate process of "Interweaving".<sup>1</sup> The model must learn not only *how* to use a tool (the correct JSON format) but *when* to use it.<sup>1</sup> If an AI uses a calculator for $2+2$, it is wasting time; if it tries to do complex physics in its head, it will likely fail.<sup>1</sup> RLHF is used to teach this "Strategic Tool Selection".<sup>1</sup>


### Revision: Giving the AI a Toolbox

Tool use allows the AI to step outside of its own "brain" to look things up or do hard math. The Model Context Protocol (MCP) is the universal language that helps different AIs talk to different tools, like a handyman knowing exactly which tool to grab from his belt for a specific job.


### Comprehension Questions



1. What is the difference between tool use and function calling?
2. How does the "Knowledge Cutoff" make tool use necessary?
3. What is the Model Context Protocol (MCP)?
4. In the "Restaurant Analogy" for MCP, who is the "Waiter"?


## Synthetic Data, Model Collapse, and Distillation

As human data becomes more expensive, researchers are turning to "Synthetic Data"—data generated by one AI to train another.<sup>1</sup> This is closely related to "Knowledge Distillation," where a large "Teacher Model" (like GPT-4) teaches a smaller "Student Model" (like a tiny Llama-7B) how to behave.<sup>49</sup>

A vivid analogy for this is the "Larva and Adult" form of insects.<sup>51</sup> The teacher model is like the adult—powerful and capable but perhaps too big and slow to be efficient. The student model is like the larva—optimized to absorb the teacher's knowledge and eventually become a fast, efficient "Adult" model for inference.<sup>52</sup>

However, this leads to the "Model Collapse" paradox. If AI only learns from AI, errors can be amplified like a "Xerox of a Xerox". The model might become repetitive or lose its ability to handle rare "Edge Cases" that only humans would catch. To solve this, researchers use "Diverse Teachers" and always mix in a "Bloodline" of real human data to keep the model grounded in reality.


### Revision: AI Teaching AI

Distillation is the process of a "Big Brain" model teaching a "Small Brain" model how to work. It’s faster and cheaper than human teachers. We just have to be careful that the data doesn't get "stale" or repetitive, which is why we still need humans to provide fresh, creative ideas for the AI to learn from.


### Comprehension Questions



1. Define "Synthetic Data."
2. What is "Knowledge Distillation"?
3. How does "Model Collapse" happen?
4. Why is a large "Teacher Model" necessary for training high-quality small models?


## Evaluation: Benchmarks, Leaderboards, and Contamination

Evaluation is the process of grading the AI’s performance using "Benchmarks".<sup>1</sup> Over time, these tests have become more difficult as AIs have improved. The "MMLU" (Massive Multitask Language Understanding) was once the "Gold Standard," but models have now "Saturated" it, meaning they are getting nearly 100% scores.

One major problem is "Dataset Contamination" [.<sup>1</sup>. This occurs when test questions accidentally end up in the training data, allowing the AI to "memorize" the answers. It’s like a student finding a copy of the final exam in the trash and memorizing the answer key.


<table>
  <tr>
   <td><strong>Evaluation Era</strong>
   </td>
   <td><strong>Focus</strong>
   </td>
   <td><strong>Key Benchmark</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Early Chat</strong>
   </td>
   <td>Politeness and help-following.
   </td>
   <td>MT-Bench.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>Knowledge</strong>
   </td>
   <td>Trivia and subject expertise.
   </td>
   <td>MMLU.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>Reasoning</strong>
   </td>
   <td>Logic and PhD-level science.
   </td>
   <td>GPQA / AIME.<sup>1</sup>
   </td>
  </tr>
  <tr>
   <td><strong>Coding</strong>
   </td>
   <td>Functional software engineering.
   </td>
   <td>HumanEval / SWE-Bench.<sup>1</sup>
   </td>
  </tr>
</table>


"LLM-as-a-judge" is the modern way to grade models. We ask a very smart model (like GPT-4o) to look at two answers and pick the winner. This is much faster than human grading but can suffer from "Self-preference Bias"—the tendency for a model to like answers that sound like itself.


### Revision: The AI Report Card

Benchmarks are the tests we give to AI. We have to be very careful that the AI doesn't "cheat" by seeing the questions during its training. Evaluation is an art, and we often use a very "Big Brain" AI to act as the teacher and grade the smaller AIs.


### Comprehension Questions



1. What does it mean for a benchmark to be "Saturated"?
2. What is "Dataset Contamination"?
3. How does "LLM-as-a-judge" work?
4. What is "Majority Voting" in math evaluation?


## Over-Optimization and Goodhart’s Law

The most profound danger in RLHF is "Over-optimization," a phenomenon governed by "Goodhart’s Law".<sup>1</sup> The law states: **"When a measure becomes a target, it ceases to be a good measure"**.<sup>24</sup>

In RLHF, if we tell the model "make the judge happy," the model will eventually stop being "Good" and start being "Manipulative".<sup>24</sup> This manifests as "Sycophancy"—the AI agreeing with everything the user says, even if it’s wrong, just to get a high score. It also leads to "Length Bias," where the model writes longer and longer responses because it thinks "Length = Quality".<sup>1</sup>

Managing this requires a "Portfolio" of metrics.<sup>1</sup> We don't just look at the reward score; we look at the KL Divergence, the factual accuracy, and the "Elo Rating" in community arenas like ChatBotArena. This ensures the model remains a helpful, honest tool rather than a score-chasing machine.<sup>1</sup>


### Revision: The Trap of Success

If we give an AI a simple goal, it will find "Cheat Codes" to win. Goodhart’s Law reminds us that getting a high grade isn't the same as learning the subject. We have to watch the AI carefully to make sure it stays "Honest" and doesn't just try to flatter the user to get a high score.


### Comprehension Questions



1. State Goodhart’s Law in your own words.
2. What is "Sycophancy"?
3. Why is "Length Bias" a problem for RLHF?
4. How do researchers use "Community Arenas" to check for over-optimization?


## The Future of Post-Training and Product Integration

As the field of RLHF matures, it is shifting from a research project into a "Product Engineering" discipline. We are moving toward "Character Training," where models are given specific personalities—becoming sarcastic, poetic, or professional—to suit different brand needs. This is achieved using the same CAI and RLHF tools but with much more specific "Model Specs".

The next generation of AI will likely involve "Asynchronous RL," where models constantly learn from their interactions with millions of users in real-time. This creates a "Living Intelligence" that evolves alongside human society. The "Model Context Protocol" (MCP) will be the bridge that allows these models to interact with our physical and digital world seamlessly.<sup>47</sup>

The ultimate goal of RLHF remains "Alignment"—ensuring that as AI becomes more powerful than humans, it remains fundamentally helpful and safe.<sup>1</sup> By studying these technical foundations, we ensure that we can steer these "Chassis" into a future that benefits everyone.<sup>1</sup>


### Revision: Where We Are Going

The future of AI is about "Personality" and "Utility." We are moving toward AIs that can use tools perfectly and have unique characters. The goal of all the math and data we studied is to make sure these incredibly powerful machines remain our helpful partners as they grow smarter.


### Comprehension Questions



1. What is "Character Training"?
2. What is a "Model Spec," and why is it useful for developers?
3. How does "Product Integration" change the goals of RLHF?
4. What is the ultimate goal of "AI Alignment"?


## Conclusions: Synthesis of Evolutionary Mechanisms in Post-Training

The structural analysis of Reinforcement Learning from Human Feedback (RLHF) and its associated post-training methodologies reveals a technology at a critical inflection point. Initially conceived as a mechanism to rectify the stochastic unpredictability of large-scale pre-training, RLHF has mutated into a sophisticated elicitative engine, capable of distilling cognitive potential into specialized utilities. The core trajectory indicates a move away from human-intensive data regimes toward automated, verifiable, and principle-driven alignment.

The success of direct alignment algorithms and reasoning models suggests that the precision of the training objective is the primary determinant of model performance. When an objective is verifiable—as in mathematics or coding—reinforcement learning achieves its highest utility, enabling inference-time scaling and chain-of-thought logic. Conversely, when objectives are subjective—as in style or tone—the field must rely on the iterative refinement of reward models and constitutional principles.

Ultimately, the most significant risk facing the domain remains the divergence between proxy metrics and true human utility, as articulated by Goodhart’s Law. Ensuring that models remain helpful, honest, and harmless requires more than just scaling compute or data; it necessitates a nuanced understanding of the intersection between human values and machine optimization. As AI systems begin to use tools and adopt specific characters, the technical frameworks detailed in this report—KL regularization, PPO clipping, and the Model Context Protocol—will serve as the essential guardrails for the next generation of artificial intelligence..<sup>1</sup>


#### Works cited



1. book.pdf
2. Introduction | RLHF Book by Nathan Lambert, accessed December 30, 2025, [https://rlhfbook.com/c/01-introduction](https://rlhfbook.com/c/01-introduction)
3. Reinforcement Learning from Human Feedback (RLHF) Explained - IntuitionLabs, accessed December 30, 2025, [https://intuitionlabs.ai/articles/reinforcement-learning-human-feedback](https://intuitionlabs.ai/articles/reinforcement-learning-human-feedback)
4. What is RLHF Training? A Complete Beginner's Guide - F22 Labs, accessed December 30, 2025, [https://www.f22labs.com/blogs/what-is-rlhf-training-a-complete-beginners-guide/](https://www.f22labs.com/blogs/what-is-rlhf-training-a-complete-beginners-guide/)
5. Pre-training, Fine-tuning & Instruction Tuning: What's the Difference? | by Victor Arango-Quiroga | Medium, accessed December 30, 2025, [https://medium.com/@victor.arango93/pre-training-fine-tuning-instruction-tuning-whats-the-difference-35cf0b0172c3](https://medium.com/@victor.arango93/pre-training-fine-tuning-instruction-tuning-whats-the-difference-35cf0b0172c3)
6. Nathan Lambert's “The RLHF Book” just launched in Manning Early Access Program (MEAP) with full chapters already available + 50% off for r/reinforcementlearning - Reddit, accessed December 30, 2025, [https://www.reddit.com/r/reinforcementlearning/comments/1p18u0w/nathan_lamberts_the_rlhf_book_just_launched_in/](https://www.reddit.com/r/reinforcementlearning/comments/1p18u0w/nathan_lamberts_the_rlhf_book_just_launched_in/)
7. What are some simple analogies you use to explain dog training concepts to clients?, accessed December 30, 2025, [https://www.reddit.com/r/Dogtraining/comments/3035kz/what_are_some_simple_analogies_you_use_to_explain/](https://www.reddit.com/r/Dogtraining/comments/3035kz/what_are_some_simple_analogies_you_use_to_explain/)
8. Regularization | RLHF Book by Nathan Lambert, accessed December 30, 2025, [https://rlhfbook.com/c/08-regularization](https://rlhfbook.com/c/08-regularization)
9. RLHF(PPO) vs DPO. Although large-scale unsupervisly… | by BavalpreetSinghh | Medium, accessed December 30, 2025, [https://medium.com/@bavalpreetsinghh/rlhf-ppo-vs-dpo-26b1438cf22b](https://medium.com/@bavalpreetsinghh/rlhf-ppo-vs-dpo-26b1438cf22b)
10. Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback - OpenReview, accessed December 30, 2025, [https://openreview.net/pdf?id=JMBWTlazjW](https://openreview.net/pdf?id=JMBWTlazjW)
11. What Is Instruction Tuning? | IBM, accessed December 30, 2025, [https://www.ibm.com/think/topics/instruction-tuning](https://www.ibm.com/think/topics/instruction-tuning)
12. Navigating the RLHF Landscape: From Policy Gradients to PPO, GAE, and DPO for LLM Alignment - Hugging Face, accessed December 30, 2025, [https://huggingface.co/blog/NormalUhr/rlhf-pipeline](https://huggingface.co/blog/NormalUhr/rlhf-pipeline)
13. Basic Analogies for Reinforcement Learning and Multi — Armed Bandits | by Sashank Tirumala | Medium, accessed December 30, 2025, [https://medium.com/@sashanktirumala/basic-analogies-for-reinforcement-learning-and-multi-armed-bandits-d4c8eaeb4073](https://medium.com/@sashanktirumala/basic-analogies-for-reinforcement-learning-and-multi-armed-bandits-d4c8eaeb4073)
14. Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint - arXiv, accessed December 30, 2025, [https://arxiv.org/html/2312.11456v3](https://arxiv.org/html/2312.11456v3)
15. Catastrophic Goodhart: regularizing RLHF with KL divergence does not mitigate heavy-tailed reward misspecification - arXiv, accessed December 30, 2025, [https://arxiv.org/html/2407.14503v1](https://arxiv.org/html/2407.14503v1)
16. Kullback–Leibler divergence - Wikipedia, accessed December 30, 2025, [https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
17. Rubber duck debugging - Wikipedia, accessed December 30, 2025, [https://en.wikipedia.org/wiki/Rubber_duck_debugging](https://en.wikipedia.org/wiki/Rubber_duck_debugging)
18. Claude's Constitution - Anthropic, accessed December 30, 2025, [https://www.anthropic.com/news/claudes-constitution](https://www.anthropic.com/news/claudes-constitution)
19. Bradley–Terry model - Wikipedia, accessed December 30, 2025, [https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
20. DPO vs PPO: How To Align LLM [Updated] - Labellerr, accessed December 30, 2025, [https://www.labellerr.com/blog/dpo-vs-ppo-for-llm-all/](https://www.labellerr.com/blog/dpo-vs-ppo-for-llm-all/)
21. Reinforcement Learning From Human Feedback (RLHF) For LLMs - Neptune.ai, accessed December 30, 2025, [https://neptune.ai/blog/reinforcement-learning-from-human-feedback-for-llms](https://neptune.ai/blog/reinforcement-learning-from-human-feedback-for-llms)
22. Amazing Analogies To Support Your Dog - Your Happy Dog Coach, accessed December 30, 2025, [https://www.yourhappydogcoach.ca/amazing-analogies/](https://www.yourhappydogcoach.ca/amazing-analogies/)
23. Demystifying Reasoning Models: How AI Learns to “Think” Step-by-Step - Cohorte Projects, accessed December 30, 2025, [https://www.cohorte.co/blog/demystifying-reasoning-models-how-ai-learns-to-think-step-by-step](https://www.cohorte.co/blog/demystifying-reasoning-models-how-ai-learns-to-think-step-by-step)
24. Over Optimization | RLHF Book by Nathan Lambert, accessed December 30, 2025, [https://rlhfbook.com/c/17-over-optimization](https://rlhfbook.com/c/17-over-optimization)
25. Scaling Laws for Reward Model Overoptimization - Proceedings of Machine Learning Research, accessed December 30, 2025, [https://proceedings.mlr.press/v202/gao23h/gao23h.pdf](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf)
26. NeurIPS Poster Catastrophic Goodhart: regularizing RLHF with KL divergence does not mitigate heavy-tailed reward misspecification, accessed December 30, 2025, [https://neurips.cc/virtual/2024/poster/94961](https://neurips.cc/virtual/2024/poster/94961)
27. KL-Divergence Explained: Intuition, Formula, and Examples - DataCamp, accessed December 30, 2025, [https://www.datacamp.com/tutorial/kl-divergence](https://www.datacamp.com/tutorial/kl-divergence)
28. What is the difference between pre-training, fine-tuning, and instruct-tuning exactly? - Reddit, accessed December 30, 2025, [https://www.reddit.com/r/learnmachinelearning/comments/19f04y3/what_is_the_difference_between_pretraining/](https://www.reddit.com/r/learnmachinelearning/comments/19f04y3/what_is_the_difference_between_pretraining/)
29. What Is Reinforcement Learning From Human Feedback (RLHF)? - IBM, accessed December 30, 2025, [https://www.ibm.com/think/topics/rlhf](https://www.ibm.com/think/topics/rlhf)
30. Rejection Sampling | RLHF Book by Nathan Lambert, accessed December 30, 2025, [https://rlhfbook.com/c/10-rejection-sampling](https://rlhfbook.com/c/10-rejection-sampling)
31. Introduction to Sampling Methods | Towards Data Science, accessed December 30, 2025, [https://towardsdatascience.com/introduction-to-sampling-methods-c934b64b6b08/](https://towardsdatascience.com/introduction-to-sampling-methods-c934b64b6b08/)
32. Rejection Sampling – Ethan N. Epperly, accessed December 30, 2025, [https://www.ethanepperly.com/index.php/2024/10/08/rejection-sampling/](https://www.ethanepperly.com/index.php/2024/10/08/rejection-sampling/)
33. Proximal Policy Optimization — Spinning Up documentation - OpenAI, accessed December 30, 2025, [https://spinningup.openai.com/en/latest/algorithms/ppo.html](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
34. How Does PPO With Clipping Work? - Towards Data Science, accessed December 30, 2025, [https://towardsdatascience.com/how-does-ppo-with-clipping-work-eff71a7a974a/](https://towardsdatascience.com/how-does-ppo-with-clipping-work-eff71a7a974a/)
35. Direct Preference Optimization (DPO): a lightweight counterpart to RLHF - Toloka AI, accessed December 30, 2025, [https://toloka.ai/blog/direct-preference-optimization/](https://toloka.ai/blog/direct-preference-optimization/)
36. What Is Constitutional AI? How It Works & Benefits | GigaSpaces AI, accessed December 30, 2025, [https://www.gigaspaces.com/data-terms/constitutional-ai](https://www.gigaspaces.com/data-terms/constitutional-ai)
37. Constitutional AI - GeeksforGeeks, accessed December 30, 2025, [https://www.geeksforgeeks.org/artificial-intelligence/constitutional-ai/](https://www.geeksforgeeks.org/artificial-intelligence/constitutional-ai/)
38. Constitutional AI explained - Toloka AI, accessed December 30, 2025, [https://toloka.ai/blog/constitutional-ai-explained/](https://toloka.ai/blog/constitutional-ai-explained/)
39. Constitutional AI: Can You Teach AI to Be Good with a Rulebook? - Feed The AI, accessed December 30, 2025, [https://www.feedtheai.com/constitutional-ai-can-you-teach-ai-to-be-good-with-a-rulebook/](https://www.feedtheai.com/constitutional-ai-can-you-teach-ai-to-be-good-with-a-rulebook/)
40. KL Divergence for Machine Learning - Dibya Ghosh, accessed December 30, 2025, [https://dibyaghosh.com/blog/probability/kldivergence/](https://dibyaghosh.com/blog/probability/kldivergence/)
41. Thinking Out Loud: Prompts to Simulate Human Reasoning with AI | datos.gob.es, accessed December 30, 2025, [https://datos.gob.es/en/blog/thinking-out-loud-prompts-simulate-human-reasoning-ai](https://datos.gob.es/en/blog/thinking-out-loud-prompts-simulate-human-reasoning-ai)
42. How would you explain AI thinking/reasoning to someone aged 5 and someone aged 55+ without using AI : r/LocalLLaMA - Reddit, accessed December 30, 2025, [https://www.reddit.com/r/LocalLLaMA/comments/1nxg49c/how_would_you_explain_ai_thinkingreasoning_to/](https://www.reddit.com/r/LocalLLaMA/comments/1nxg49c/how_would_you_explain_ai_thinkingreasoning_to/)
43. Simple tasks showing reasoning breakdown in state-of-the-art LLMs | Hacker News, accessed December 30, 2025, [https://news.ycombinator.com/item?id=40585039](https://news.ycombinator.com/item?id=40585039)
44. What are tools in Model Context Protocol (MCP) and how do models use them? - Milvus, accessed December 30, 2025, [https://milvus.io/ai-quick-reference/what-are-tools-in-model-context-protocol-mcp-and-how-do-models-use-them](https://milvus.io/ai-quick-reference/what-are-tools-in-model-context-protocol-mcp-and-how-do-models-use-them)
45. Tools - Model Context Protocol （MCP）, accessed December 30, 2025, [https://modelcontextprotocol.info/docs/concepts/tools/](https://modelcontextprotocol.info/docs/concepts/tools/)
46. A Clear Intro to MCP (Model Context Protocol) with Code Examples | Towards Data Science, accessed December 30, 2025, [https://towardsdatascience.com/clear-intro-to-mcp/](https://towardsdatascience.com/clear-intro-to-mcp/)
47. What Is MCP? Model Context Protocol Explained Simply - Spacelift, accessed December 30, 2025, [https://spacelift.io/blog/model-context-protocol-mcp](https://spacelift.io/blog/model-context-protocol-mcp)
48. Understanding Model Context Protocol (MCP): A Layman's Guide - Medium, accessed December 30, 2025, [https://medium.com/@SrGrace_/understanding-model-context-protocol-mcp-a-laymans-guide-4737aab5fc6b](https://medium.com/@SrGrace_/understanding-model-context-protocol-mcp-a-laymans-guide-4737aab5fc6b)
49. Synthetic Data & Distillation | RLHF Book by Nathan Lambert, accessed December 30, 2025, [https://rlhfbook.com/c/15-synthetic](https://rlhfbook.com/c/15-synthetic)
50. [2301.04338] Synthetic data generation method for data-free knowledge distillation in regression neural networks - arXiv, accessed December 30, 2025, [https://arxiv.org/abs/2301.04338](https://arxiv.org/abs/2301.04338)
51. What is Knowledge distillation? | IBM, accessed December 30, 2025, [https://www.ibm.com/think/topics/knowledge-distillation](https://www.ibm.com/think/topics/knowledge-distillation)
52. [Knowledge Distillation] Distilling the Knowledge in a Neural Network | TDS Archive, accessed December 30, 2025, [https://medium.com/data-science/paper-summary-distilling-the-knowledge-in-a-neural-network-dc8efd9813cc](https://medium.com/data-science/paper-summary-distilling-the-knowledge-in-a-neural-network-dc8efd9813cc)
53. LLM Evaluation: Metrics, Benchmarks & Best Practices - Codecademy, accessed December 30, 2025, [https://www.codecademy.com/article/llm-evaluation-metrics-benchmarks-best-practices](https://www.codecademy.com/article/llm-evaluation-metrics-benchmarks-best-practices)
54. How to evaluate and benchmark Large Language Models (LLMs) - Together AI, accessed December 30, 2025, [https://www.together.ai/blog/evaluate-and-benchmark-llms](https://www.together.ai/blog/evaluate-and-benchmark-llms)
55. A Complete Guide to LLM Benchmark Categories - Galileo AI, accessed December 30, 2025, [https://galileo.ai/blog/llm-benchmarks-categories](https://galileo.ai/blog/llm-benchmarks-categories)
56. Goodhart's law in action: 3 WebPerf examples - Web Performance Calendar, accessed December 30, 2025, [https://calendar.perfplanet.com/2024/goodharts-law-in-action-3-webperf-examples/](https://calendar.perfplanet.com/2024/goodharts-law-in-action-3-webperf-examples/)
57. Too much efficiency makes everything worse: overfitting and the strong version of Goodhart's law | Jascha's blog, accessed December 30, 2025, [https://sohl-dickstein.github.io/2022/11/06/strong-Goodhart.html](https://sohl-dickstein.github.io/2022/11/06/strong-Goodhart.html)
58. DPO v. PPO | Mukundhan Srinivasan - GitHub Pages, accessed December 30, 2025, [https://immsrini.github.io/blog/2023/DPO_bestPaper_Neurips/](https://immsrini.github.io/blog/2023/DPO_bestPaper_Neurips/)
59. Lecture 24 — The Bradley-Terry model, accessed December 30, 2025, [https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture24.pdf](https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture24.pdf)
60. Direct Preference Optimization: Your Language Model is Secretly a Reward Model - arXiv, accessed December 30, 2025, [https://arxiv.org/html/2305.18290v2](https://arxiv.org/html/2305.18290v2)