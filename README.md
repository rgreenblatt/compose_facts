I've removed the composition dataset to reduce leakage, but you can regenerate it using `python3 create_compositional_dataset.py -n 300 && python3 create_compositional_dataset.py -n 300 --element-names`. PLEASE DON'T PUBLICLY POST THIS DATASET INCLUDING BY PUSHING IT TO GITHUB. (I also removed the correct answer in the "run_manifold_eval.py" file, you can manually edit this back in to run this file.)

See write_up.md for details. I discuss model performance on multi-hop latent reasoning tasks in "[Recent LLMs can do 2-hop and 3-hop latent (no-CoT) reasoning on natural facts](https://www.lesswrong.com/posts/aYtrLhoZtCKZnfBvA/recent-llms-can-do-2-hop-and-3-hop-latent-no-cot-reasoning)".

---

To replicate specifically the results on the exact question used for Leo Gao's manifold markets, run:

- For the easier version of the question: `python3 run_manifold_eval.py --k-shot=0 -f 300 -n 1 -t 0.0 -m opus-4-5` 
- For the harder version of the question: `python3 run_manifold_eval.py --k-shot=0 -f 300 --harder -n 1 --long-prefill -t 0.0 -m gemini-3-pro` 

You will first need to manually edit back in "correct_answer" (see file for details).

Note that Gemini 3 Pro is very fiddly about whether or not prefill actually works to prevent it from reasoning on a given prompt meaning that many close by prompts might result in the model repeatedly reasoning and then this error-ing out after retries.

