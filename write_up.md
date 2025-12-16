# Frontier LLMs can compose facts without chain-of-thought

In January 2023, Leo Gao hypothesized that LLMs would struggle to compose facts without Chain-of-Thought (CoT).
He operationalized this with a [manifold question](https://manifold.markets/LeoGao/will-a-big-transformer-lm-compose-t) about whether LLMs would be able to answer "What is the sum of the atomic number of uranium and the age at which Euler died?" without Chain-of-Thought (CoT) by 2026.
Opus 4.5 reliably (128/128) answers this question correctly (at t=1.0) and with some prompting tricks Opus 4 and Sonnet 4 also answer this question correctly all of the time.[^repeat]
Additionally, I find that on a dataset of similar fact-composition questions, Opus 4.5 gets 85.7% correct (64.7% without few-shot prompting and without prompting tricks). 

[^repeat]: The prompting trick is to repeat the problem 5 times as I discuss below. Using a 10 shot prompt with no repeats also yields perfect performance for Opus 4 and Sonnet 4.

In a prior post, I found that repeating the question multiple times improves no-CoT performance for math questions.
I find that this also improves performance on this fact-composition dataset.
I find that 5 repeats seems to work best (similar to math results).
Interestingly, if I use a 0-shot prompt (instead of the 10-shot prompt I use by default), repeating the question 5 times boost performance all the way from 64.7% to 84.7% for Opus 4.5, implying that repeats can substitute for a few-shot prompt in this context. (I see a similarly large boost for Sonnet 4).

Here are the results with a 10-shot prompt on this dataset:

![](https://raw.githubusercontent.com/rgreenblatt/compose_facts/master/eval_results/bar_chart.png)

And with either 0-shot or 10-shot:

![](https://raw.githubusercontent.com/rgreenblatt/compose_facts/master/eval_results/zero_shot_comparison.png)

(Sonnet 4 actually does better with 0-shot + repeats than with 10-shot + repeats.)

The public version of this code base can be found [here](https://github.com/rgreenblatt/compose_facts). I've removed the composition dataset to reduce leakage, but you can regenerate it using `python3 create_compositional_dataset.py -n 300`. PLEASE DON'T PUBLICLY POST THIS DATASET INCLUDING BY PUSHING IT TO GITHUB. (I also removed the correct answer in the "run_manifold_eval.py" file, you can manually edit this back in to run this file.)

# Dataset details 

I made the overall dataset by composing facts from 3 datasets of easy trivia questions:

- **Ages**: How old was this famous person when they died?
- **Atomic**: What is the atomic number of element X?
- **Static**: A list of questions with an integer answer written by Opus 4.5 that another copy of Opus 4.5 got right.

The models get 100% of these datasets correct without CoT (except for the ages dataset on which they only get 98% correct).

Then, I compose these facts by asking for the sum of the two numbers like "What is the sum of the number of essays in The Federalist Papers and the atomic number of moscovium?".

Here are 15 random questions from this dataset:

- What is the sum of the number of constellations officially recognized by the IAU and the age at which Louis Philippe I died?
- What is the sum of the number of vertebrae in the human spine and the atomic number of gold?
- What is the sum of the number of movements in Messiaen's Vingt regards sur l'enfant-Jésus and the atomic number of copper?
- What is the sum of the age at which Mark Twain died and the atomic number of seaborgium?
- What is the sum of the age at which Pyotr Ilyich Tchaikovsky died and the number of violin concertos Vivaldi composed?
- What is the sum of the atomic number of promethium and the age at which Ingmar Bergman died?
- What is the sum of the number of peaks over 8000 meters and the atomic number of antimony?
- What is the sum of the age at which Louis XIV of France died and the atomic number of holmium?
- What is the sum of the number of verses in the Tao Te Ching and the atomic number of tennessine?
- What is the sum of the atomic number of oganesson and the number of moons Saturn has?
- What is the sum of the number of moons Saturn has and the atomic number of scandium?
- What is the sum of the number of cello suites Bach composed and the atomic number of rhenium?
- What is the sum of the atomic number of bromine and the age at which Alfred Jarry died?
- What is the sum of the atomic number of iron and the number of movements in Messiaen's Vingt regards sur l'enfant-Jésus?
- What is the sum of the number of floors in the Empire State Building and the atomic number of gallium?

Note that some of these facts might be sensitive to the data cutoff of the model, I haven't really tried seriously to avoid this.

For the exact dataset and more details, see the open source repo [here](https://github.com/rgreenblatt/compose_facts).

# Appendix: How helpful were AIs for this project?

I tried to get Opus 4.5 to do this entire project end-to-end after giving it a description and a chance to ask clarifying questions.
It seemed moderately close to doing a decent job on the empirical side of the project, but in practice there were several issues that I ended up correcting:

- The initial dataset the model made wasn't ideal in a few ways (mostly the "static" questions it generated were often overly easy IMO and the code it used for combining the facts was written strangely and only used certain orderings of the pairings of different datasets).
- The model didn't do a good job sanity checking itself (e.g. looking at the dataset it generated) by default without reminders, but with reminders it did a OK job (though missed some stuff I wanted to change) and ended up correcting some bugs. (In another similar project, I found it missed a fatal issue that was easy to notice after it did a sanity check on the dataset at my urging but immediately noticed and fixed the issue after I asked it if it noticed anything wrong.)
- The model made somewhat strange/bad choices on how to handle the few-shot prompt intersecting with the question the model was being evaluated on. This resulted in a worse evaluation and messier code than what I got after reprogramming it. (I explicitly told it to handle overlap, I'm not sure it would have handled this at all by default.)
- Generally the model seemed somewhat lazy and it did the most minimal thing rather than (e.g.) generating reasonable plots without prompting.
- I think the model would have made bad choices about exactly what experiments to run and not run and might have missed some interesting findings.

By the end of the project, I managed the AI for the last bit in an extremely hands on way (or did it myself), but it did the bulk of the project autonomously with moderate management (still much more management than I would give an intern).
This write up is entirely written by me.

Generally, I think an intern could probably have done a decent job with this project without almost any management from me and Opus 4.5 underperformed this baseline by a decent amount, though for this small project it doesn't seem super far away.

# Appendix: full result tables

| Model | r=1 | r=5 | Δ |
|-------|-----|-----|---|
| Haiku 3.5 | 2.0% | 2.3% | +0.3% |
| Sonnet 4 | 75.0% | 75.7% | +0.7% |
| Opus 4 | 81.3% | 84.3% | +3.0% |
| Opus 4.5 | 81.0% | 85.7% | +4.7% |


| Model | k=0, r=1 | k=0, r=5 | k=10, r=1 | k=10, r=5 |
|-------|----------|----------|-----------|-----------|
| Sonnet 4 | 63.7% | 80.3% | 75.0% | 75.7% |
| Opus 4.5 | 64.7% | 84.7% | 81.0% | 85.7% |
