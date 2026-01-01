# Frontier LLMs can compose facts without chain-of-thought

In January 2023, Leo Gao hypothesized that future LLMs would continue to struggle to compose facts without Chain-of-Thought (CoT).
He operationalized this with a [Manifold question](https://manifold.markets/LeoGao/will-a-big-transformer-lm-compose-t) about whether LLMs would be able to answer "What is the sum of the atomic number of uranium and the age at which Euler died?" without Chain-of-Thought (CoT) by 2026.
Opus 4.5 and Gemini 3 Pro both reliably (128/128) answer this question correctly (at t=1.0) and with either a  10-shot prompt or if we repeat the problem 5 times, Opus 4 and Sonnet 4 also answer this question correctly all of the time.[^promptingdesc]
He also operationalized a harder test: whether LLMs can answer "What is the name of the element with an atomic number equal to the sum of the age at which Euler died and the number of faces on a cube?".
I find that with filler tokens, Gemini 3 Pro can answer this question 80% of the time (without filler tokens, but still with 20-shot prompting it gets it right 20% of the time).[^temperature]
Opus 4 and Opus 4.5 always get this question wrong at t=1.0 (0/128).

[^promptingdesc]: Opus 4.5 doesn't require a k-shot prompt or repeats to consistently get this right. I always use a 20-shot prompt for Gemini and Gemini doesn't need anything else to get this problem right.

See [my earlier post about filler tokens and repeats](https://www.lesswrong.com/posts/NYzYJ2WoB74E6uj9L/recent-llms-can-use-filler-tokens-or-problem-repeats-to) for discussion of the effects of filler tokens and repeats (and the prompting setup I use).


[^temperature]: The original market asked about the response at t=0. However, if Gemini is evaluated using my prompt at t=0, I find that the model consistently reasons which invalidates the measurement. I do find that at t=0.3 (and resampling until the model no longer reasons) the model gets it right 8/8 times.

I also construct a dataset of similar fact-composition questions to these questions.
The easier version of this dataset (corresponding to the easier question) consists of the form "fact1 + fact2" where facts can be of the form "at what age did X die", "what is the atomic number of Y", or selected from a distribution of knowledge questions with an integer answer less than 1000 (e.g. "What is the number of plays Shakespeare wrote?").[^knowledge]
I also test a generalized version of this where we add the result of N 1-hop questions in "Appendix: adding the result of N 1-hop questions".
The harder version of this dataset (corresponding to the harder question) is of the form "What is the name of the element with an atomic number equal to fact1 + fact2" using the same distribution of facts as the easier dataset, except that I exclude atomic number facts.

On the easier version of this dataset, Opus 4.5 gets 85.7% correct with 5 problem repeats (64.7% without few-shot prompting and without repeats) and Gemini 3 Pro gets 89.0% correct with filler tokens counting to 300 (86.7% without filler tokens, but still with 20-shot prompting).
For specifically questions of the form "atomic number of X + age at which Y died", Gemini 3 Pro gets 91.5% correct and Opus 4.5 gets 85.1% indicating that the other types of questions are at a similar level of difficulty.

On the harder version of this dataset, Opus 4.5 gets 7.3% correct (with 5 problem repeats) and Gemini 3 Pro gets 36.0% correct (with filler tokens counting to 300).
Qualitatively, the harder version of my dataset is somewhat harder than Leo Gao's harder question (because "the number of faces on a cube" is particularly small and easy).
So, Gemini 3 Pro getting that question right ~80% of the time is consistent with the model getting a bit lucky or with that question being a bit easier than the typical question on a distribution where the model gets 36.0% correct.

[^knowledge]: I include all combinations except age plus age.

Thus, I overall think Leo's prediction was wrong and both markets should resolve to yes: LLMs can now compose facts well enough to get both of these questions right and it looks like performance on latent multi-hop reasoning has been improving and will continue to improve.

Code for these experiments can be found at [github.com/rgreenblatt/compose_facts](https://github.com/rgreenblatt/compose_facts). I've removed the datasets to reduce leakage, but you can regenerate them using `python3 create_compositional_dataset.py -n 300 && python3 create_compositional_dataset.py -n 300 --element-names`. PLEASE DON'T PUBLICLY POST THIS DATASET INCLUDING BY PUSHING IT TO GITHUB. (I also removed the correct answer in the "run_manifold_eval.py" file, you can manually edit this back in to run this file.)

# The effect of problem repeats and few-shots on these fact composition questions

For Anthropic models, I find that 5 repeats seems to work best (similar to what I found in [my earlier post about filler tokens](https://www.lesswrong.com/posts/NYzYJ2WoB74E6uj9L/recent-llms-can-use-filler-tokens-or-problem-repeats-to) for math questions).
For Gemini 3 Pro, I find that the model very often responds invalidly when I use repeats or when not using filler tokens on the harder version of this dataset.
Thus, this section will just discuss results on Anthropic models.

Interestingly, if I use a 0-shot prompt (instead of the 10-shot prompt I use by default), then on the easier version of the dataset, repeating the question 5 times boosts performance all the way from 64.7% to 84.7% for Opus 4.5, implying that repeats can substitute for a few-shot prompt in this context. (I see a similarly large boost for Sonnet 4.) I don't know why this is the case.

Here are the results with a 10-shot prompt on the numerical answer dataset:

![](https://raw.githubusercontent.com/rgreenblatt/compose_facts/master/eval_results/bar_chart.png)

And with either 0-shot or 10-shot:

![](https://raw.githubusercontent.com/rgreenblatt/compose_facts/master/eval_results/zero_shot_comparison.png)

(Sonnet 4 actually does better with 0-shot + repeats than with 10-shot + repeats.)

And the plots for the version of the dataset where the answer is converted to the name of an element:

![](https://raw.githubusercontent.com/rgreenblatt/compose_facts/master/eval_results/element_bar_chart.png)

![](https://raw.githubusercontent.com/rgreenblatt/compose_facts/master/eval_results/element_zero_shot_comparison.png)


# Dataset details 

I made the easier dataset by composing facts from 3 datasets of easy trivia questions:

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

For the harder dataset, I sample from age+age, age+static, or static+static and convert the output to an element number.
We reject problems that produce a number greater than 108 (to avoid invalid numbers while also discarding particularly obscure elements).

Here are 15 random questions from this dataset:

- What is the name of the element with an atomic number equal to the sum of the number of tarot cards in the major arcana and the number of counties in England?
- What is the name of the element with an atomic number equal to the sum of the number of cello suites Bach composed and the number of dots on a standard die?
- What is the name of the element with an atomic number equal to the sum of the age at which Michael Faraday died and the number of eclogues Virgil wrote?
- What is the name of the element with an atomic number equal to the sum of the number of arches in the Colosseum's outer wall and the number of books in Homer's Iliad?
- What is the name of the element with an atomic number equal to the sum of the number of bones in the human skull and the number of prefectures in Japan?
- What is the name of the element with an atomic number equal to the sum of the number of books in the Old Testament and the number of tales in The Canterbury Tales?
- What is the name of the element with an atomic number equal to the sum of the number of vertebrae in the human spine and the number of operas Wagner composed?
- What is the name of the element with an atomic number equal to the sum of the number of books in Livy's Ab Urbe Condita that survive and the number of dominoes in a double-six set?
- What is the name of the element with an atomic number equal to the sum of the number of operas Wagner composed and the number of constellations officially recognized by the IAU?
- What is the name of the element with an atomic number equal to the sum of the age at which Franz Schubert died and the age at which Marcel Proust died?
- What is the name of the element with an atomic number equal to the sum of the number of letters in the Thai alphabet and the number of plays Shakespeare wrote?
- What is the name of the element with an atomic number equal to the sum of the number of plays Shakespeare wrote and the number of études Ligeti composed for piano?
- What is the name of the element with an atomic number equal to the sum of the number of nocturnes Chopin composed and the number of verses in the Tao Te Ching?
- What is the name of the element with an atomic number equal to the sum of the number of players on a cricket team and the number of partitas for solo violin Bach composed?
- What is the name of the element with an atomic number equal to the sum of the number of eclogues Virgil wrote and the number of moons Jupiter has?

For the exact dataset and more details, see the open source repo [here](https://github.com/rgreenblatt/compose_facts).

# Appendix: How helpful were AIs for this project?

I tried to get Opus 4.5 to do this entire project end-to-end after giving it a description and a chance to ask clarifying questions.
It seemed moderately close to doing a decent job on the empirical side of the project, but in practice there were several issues that I ended up correcting:

- The initial dataset the model made wasn't ideal in a few ways (mostly the "static" questions it generated were often overly easy in my opinion and the code it used for combining the facts was written strangely and only used certain orderings of the pairings of different datasets).
- The model didn't do a good job sanity checking itself (e.g., looking at the dataset it generated) by default without reminders, but with reminders it did an OK job (though missed some stuff I wanted to change) and ended up correcting some bugs. (In another similar project, I found it missed a fatal issue that was easy to notice after it did a sanity check on the dataset at my urging but immediately noticed and fixed the issue after I asked it if it noticed anything wrong.)
- The model made somewhat strange/bad choices on how to handle the few-shot prompt intersecting with the question the model was being evaluated on. This resulted in a worse evaluation and messier code than what I got after reprogramming it. (I explicitly told it to handle overlap; I'm not sure it would have handled this at all by default.)
- Generally the model seemed somewhat lazy and it did the most minimal thing rather than (e.g.) generating reasonable plots without prompting.
- I tend to be pickier than the model with respect to experiment choices and getting things exactly right.
- I think the model would have made bad choices about exactly what experiments to run and not run and might have missed some interesting findings if I didn't experiment around some. It generally doesn't seem to do open-ended experimentation.
- On a few things, I thought the model would probably make a bad choice, so I specified exactly what I wanted (but maybe I'm wrong and the model would have done something good).

By the end of the project, I managed the AI for the last bit in an extremely hands-on way (or did it myself), but it did the bulk of the project autonomously with moderate management (still much more management than I would give an intern).
This write-up is entirely written by me (and I'd guess that Opus 4.5 wouldn't have done a satisfactory job with the write-up).

Generally, I think an intern could probably have done a decent job with this project without almost any management from me and Opus 4.5 underperformed this baseline by a decent amount, though for this small project it doesn't seem super far away.

# Appendix: full result tables

## Easier version

| Model | r=1 | r=5 | Δ |
|-------|-----|-----|---|
| Haiku 3.5 | 2.0% | 2.3% | +0.3% |
| Sonnet 4 | 75.0% | 75.7% | +0.7% |
| Opus 4 | 81.3% | 84.3% | +3.0% |
| Opus 4.5 | 81.0% | 85.7% | +4.7% |


| Model | k=0, r=1 | k=0, r=5 | k=10, r=1 | k=10, r=5 |
|-------|----------|----------|-----------|-----------|
| Sonnet 4 | 63.7% | 80.3% | 75.0% | 75.7% |
| Opus 4 | 56.3% | 83.0% | 81.3% | 84.3% |
| Opus 4.5 | 64.7% | 84.7% | 81.0% | 85.7% |

## Output as an element name

| Model | r=1 | r=5 | Δ |
|-------|-----|-----|---|
| Haiku 3.5 | 3.0% | 2.3% | -0.7% |
| Sonnet 4 | 2.0% | 2.3% | +0.3% |
| Opus 4 | 8.7% | 9.3% | +0.7% |
| Opus 4.5 | 3.0% | 7.3% | +4.3% |

| Model | k=0, r=1 | k=0, r=5 | k=10, r=1 | k=10, r=5 |
|-------|----------|----------|-----------|-----------|
| Sonnet 4 | 1.0% | 1.3% | 2.0% | 2.3% |
| Opus 4 | 1.3% | 3.3% | 8.7% | 9.3% |
| Opus 4.5 | 1.7% | 5.3% | 3.0% | 7.3% |

