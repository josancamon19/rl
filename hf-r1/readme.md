| Step | Process     | Description                                                                                                                                                               |
| ---- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Observation | The agent observes the environment. The agent takes in information about its current state and surroundings.                                                              |
| 2    | Action      | The agent takes an action based on its current policy. Using its learned strategy (policy), the agent decides what to do next.                                            |
| 3    | Feedback    | The environment gives the agent a reward. The agent receives feedback on how good or bad its action was.                                                                  |
| 4    | Learning    | The agent updates its policy based on the reward. The agent adjusts its strategy—reinforcing actions that led to high rewards and avoiding those that led to low rewards. |
| 5    | Iteration   | Repeat the process. This cycle continues, allowing the agent to continuously improve its decision-making.                                                                 |


| Think about learning to ride a bike. You might wobble and fall at first (negative reward!). But when you manage to balance and pedal smoothly, you feel good (positive reward!). You adjust your actions based on this feedback – leaning slightly, pedaling faster, etc. – until you learn to ride well. RL is similar – it’s about learning through interaction and feedback.



| Benefit                              | Description                                                                                                                                                                                                                                                          |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Improved Control                     | RL allows us to have more control over the kind of text LLMs generate. We can guide them to produce text that is more aligned with specific goals, like being helpful, creative, or concise.                                                                         |
| Enhanced Alignment with Human Values | RLHF, in particular, helps us align LLMs with complex and often subjective human preferences. It’s hard to write down rules for “what makes a good answer,” but humans can easily judge and compare responses. RLHF lets the model learn from these human judgments. |
| Mitigating Undesirable Behaviors     | RL can be used to reduce negative behaviors in LLMs, such as generating toxic language, spreading misinformation, or exhibiting biases. By designing rewards that penalize these behaviors, we can nudge the model to avoid them.                                    |


Next token predictions fail short.

RLHF: Human prefs, train reward model, fine tune with RL (PPO), where policy is llm itself, value model is the reward model.

GRPO does not use preference data like DPO, but instead compares groups of similar samples using a reward signal from a model or function.

GRPO can incorporate reward signals from any function or model that can evaluate the quality of responses.


-----


R1 Aha moment

1. Initial Attempt: The model makes an initial attempt at solving a problem
2. Recognition: It recognizes potential errors or inconsistencies
3. Self-Correction: It adjusts its approach based on this recognition
4. Explanation: It can explain why the new approach is better


The Reasoning RL Phase focuses on developing core reasoning capabilities across domains including mathematics, coding, science, and logic. This phase employs rule-based reinforcement learning, with rewards directly tied to solution correctness.

Crucially, all the tasks in this phase are ‘verifiable’

The final Diverse RL Phase tackles multiple task types using a sophisticated hybrid approach. For deterministic tasks, it employs rule-based rewards, while subjective tasks are evaluated through LLM feedback. This phase aims to achieve human preference alignment through its innovative hybrid reward approach, combining the precision of rule-based systems with the flexibility of language model evaluation.

The first step in GRPO is remarkably intuitive - it’s similar to how a student might solve a difficult problem by trying multiple approaches. When given a prompt, the model doesn’t just generate one response; instead, it creates multiple attempts at solving the same problem (usually 4, 8, or 16 different attempts).


This is where GRPO really shines in its simplicity. Unlike other methods for RLHF that need always require a separate reward model to predict how good a solution might be, GRPO can use any function or model to evaluate the quality of a solution. For example, we could use a length function to reward shorter responses or a mathematical solver to reward accurate mathematical solutions.


`Advantage = (reward - mean(group_rewards)) / std(group_rewards)`


Optimization: Learning from Experience
The final step is where GRPO teaches the model to improve based on what it learned from evaluating the group of solutions. This process is both powerful and stable, using two main principles:

It encourages the model to produce more solutions like the successful ones while moving away from less effective approaches
It includes a safety mechanism (called KL divergence penalty) that prevents the model from changing too drastically all at once


```text
Input: 
- initial_policy: Starting model to be trained
- reward_function: Function that evaluates outputs
- training_prompts: Set of training examples
- group_size: Number of outputs per prompt (typically 4-16)

Algorithm GRPO:
1. For each training iteration:
   a. Set reference_policy = initial_policy (snapshot current policy)
   b. For each prompt in batch:
      i. Generate group_size different outputs using initial_policy
      ii. Compute rewards for each output using reward_function
      iii. Normalize rewards within group:
           normalized_advantage = (reward - mean(rewards)) / std(rewards)
      iv. Update policy by maximizing the clipped ratio:
          min(prob_ratio * normalized_advantage, 
              clip(prob_ratio, 1-epsilon, 1+epsilon) * normalized_advantage)
          - kl_weight * KL(initial_policy || reference_policy)
          
          where prob_ratio is current_prob / reference_prob

Output: Optimized policy model
```



----
Step 1: Group Sampling
For each questiong, the model will generate G outputs (group size) from the trained policy: {
01,02,03,. .., OGT0a },G = 8 where eacho; represents one completion from the model.

Step 2: Advantage Calculation
Once we have multiple responses, we need a way to determine which ones are better than others

Continuing with our arithmetic example for the same example above, imagine we have 8 responses, 4 of which is correct and the rest wrong, therefore;

| Metric                        | Value                                 |
|-------------------------------|---------------------------------------|
| Group Average                 | mean(ri) = 0.5                        |
| Standard Deviation            | std(ri) = 0.53                        |
| Advantage (Correct response)  | Ai = (1 - 0.5) / 0.53 = 0.94          |
| Advantage (Wrong response)    | Ai = (0 - 0.5) / 0.53 = -0.94         |

Step 3: Policy Update
The final step is to use these advantage values to update our model so that it becomes more likely to generate good responses in the future.

Probability ratio + Clip function + KL Divergence