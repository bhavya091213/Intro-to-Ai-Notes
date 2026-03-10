# Bayesian networks


### RECAP
- Gemini recap each one of these by giving university level notes and breakdowns and equations and tables. Use cases, where it benifits, pitfalls: Marignal distributions, and Joint distributions and conditional probability. Use markdown formatting
  - Gemini response:
    ## Bayesian Networks: A University-Level Recap

    Bayesian networks (BNs) are probabilistic graphical models that represent a set of random variables and their conditional dependencies via a directed acyclic graph (DAG).

    ### Key Concepts and Components:

    1.  **Directed Acyclic Graph (DAG):**
        *   **Nodes:** Represent random variables (e.g., diseases, symptoms, events).
        *   **Edges (Arrows):** Represent direct probabilistic dependencies between variables. An arrow from variable A to variable B implies that A has a direct influence on B.
        *   **Acyclicity:** Ensures that there are no directed cycles, meaning you cannot start at a node and follow the arrows to return to the same node. This is crucial for defining a consistent probability distribution.

    2.  **Conditional Probability Distributions (CPDs):**
        *   Each node in the DAG has an associated CPD that quantizes the probabilistic relationship between that node and its parents.
        *   For a node $\text{X}$ with parents $P_1, P_2, \dots, P_n$, its CPD is $P(\text{X} | P_1, P_2, \dots, P_n)$.
        *   If a node has no parents (a root node), its CPD is simply its marginal probability distribution, $P(\text{X})$.

    ### Core Probabilistic Concepts:

    *   **Joint Probability Distribution:**
        *   The probability of a specific combination of all variables in the network occurring together.
        *   For a BN with variables $X_1, X_2, \dots, X_n$, the joint probability distribution is given by the chain rule:
            $$P(X_1, X_2, \dots, X_n) = \prod_{i=1}^{n} P(X_i | \text{Parents}(X_i))$$ 
        *   This factorization is the core power of BNs; it decomposes a potentially complex joint distribution into a product of simpler conditional probabilities, exploiting conditional independencies.

    *   **Marginal Probability Distribution:**
        *   The probability of a single variable occurring, irrespective of the values of other variables.
        *   It's obtained by summing (or integrating, for continuous variables) the joint probability distribution over all other variables.
        *   $$P(X_i) = \sum_{X_1, \dots, X_{i-1}, X_{i+1}, \dots, X_n} P(X_1, X_2, \dots, X_n)$$ 
        *   In practice, exact marginalization can be computationally expensive for large networks. Algorithms like Variable Elimination or sampling methods (e.g., MCMC) are used.

    *   **Conditional Probability:**
        *   The probability of an event occurring given that another event has already occurred.
        *   In BNs, this is directly represented by the CPDs. For example, the probability of a symptom (e.g., fever) given a disease (e.g., flu) is $P(\text{Fever} | \text{Flu})$.
        *   **Bayes' Theorem** is fundamental for inference, allowing us to update beliefs about causes given evidence of effects:
            $$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$
            Where $P(A|B)$ is the posterior probability, $P(B|A)$ is the likelihood, $P(A)$ is the prior probability, and $P(B)$ is the evidence.

    ### Tables for CPDs (Example: Node 'Fever' with parents 'Flu' and 'Cold'):

    | Flu   | Cold  | P(Fever=True | Flu, Cold) | P(Fever=False | Flu, Cold) |
    | :---- | :---- | :------------------------- | :-------------------------- |
    | True  | True  | 0.9                        | 0.1                         |
    | True  | False | 0.7                        | 0.3                         |
    | False | True  | 0.6                        | 0.4                         |
    | False | False | 0.1                        | 0.9                         |

    Assuming prior probabilities for Flu and Cold:
    $P(\text{Flu=True}) = 0.2$, $P(\text{Flu=False}) = 0.8$
    $P(\text{Cold=True}) = 0.3$, $P(\text{Cold=False}) = 0.7$

    ### Use Cases:

    *   **Medical Diagnosis:** Inferring the probability of diseases given symptoms.
    *   **Spam Filtering:** Classifying emails based on the presence of certain words.
    *   **Risk Assessment:** Evaluating financial or security risks based on various factors.
    *   **Computer Vision:** Object recognition and scene understanding.
    *   **Genetics:** Modeling gene interactions.

    ### Benefits:

    *   **Interpretability:** The graphical structure provides an intuitive understanding of relationships between variables.
    *   **Efficiency:** The factorization of the joint distribution makes inference computationally more tractable than dealing with a full joint probability table.
    *   **Knowledge Representation:** Can incorporate expert knowledge easily.
    *   **Handling Uncertainty:** Explicitly models probabilistic relationships.
    *   **Causal Reasoning:** Can be designed to represent causal relationships (though not all BNs are causal).

    ### Pitfalls:

    *   **Computational Complexity:** Exact inference can still be NP-hard in general graphs, especially with many variables or complex dependencies.
    *   **Parameter Learning:** Estimating CPDs from data requires sufficient data, and learning can be challenging with sparse data.
    *   **Structure Learning:** Determining the optimal graph structure from data is also a complex problem.
    *   **Assumption of DAG:** The acyclic nature might not always perfectly reflect real-world feedback loops.
    *   **Independence Assumptions:** The model relies heavily on the conditional independence assumptions encoded in the graph; incorrect assumptions can lead to inaccurate results.

    ### Inference:

    *   **Inference** in BNs refers to computing probabilities of interest, such as marginals or posteriors, given some evidence (observed variables).
    *   **Exact Inference:** Algorithms like Variable Elimination and Junction Tree algorithm guarantee correct probabilities but can be exponential in complexity.
    *   **Approximate Inference:** Methods like Markov Chain Monte Carlo (MCMC) sampling (e.g., Gibbs sampling) and Variational Inference are used when exact inference is infeasible, providing approximations to the true probabilities.
  

## Product Rule
- SOemtiems, cond distributions, want join from there
- P(y)P(x|y). P(x,y)
- Get join giving marginal or conditional
- Chain rule
  - Always write joint distribution, incremental product of conditional distributions
  - Gemini give somee quation breakdowns for this
    - Gemini response:
      The joint distribution of a Bayesian Network can be expressed as the product of conditional probability distributions of each variable given its parents. This is represented by the chain rule: $P(X_1, ..., X_n) = \prod_{i=1}^{n} P(X_i | 	ext{Parents}(X_i))$. For example, if a network has variables A, B, and C, and B and C are parents of A, the joint distribution is $P(A, B, C) = P(A|B, C)P(B)P(C)$.
  - That was for 3 variables
  - We can extend to N variables via the big PI notation, product of all cond probs
- Gemini give a breakdown on Bayes Theorm/rule and have a small worked out example
  - Gemini response:
    Bayes' Theorem, also known as Bayes' rule, is a fundamental concept in probability theory that describes how to update the probability of a hypothesis based on new evidence. It is mathematically expressed as:

    $P(H|E) = \frac{P(E|H)P(H)}{P(E)}$

    Where:
    - $P(H|E)$ is the **posterior probability**: the probability of the hypothesis (H) given the evidence (E).
    - $P(E|H)$ is the **likelihood**: the probability of the evidence (E) given that the hypothesis (H) is true.
    - $P(H)$ is the **prior probability**: the initial probability of the hypothesis (H) before observing any evidence.
    - $P(E)$ is the **marginal probability of the evidence**: the overall probability of the evidence (E) occurring, regardless of the hypothesis.
  - Class example: P(+m) * P(+s|+m) / P(+s)
    - We don't have the probability of +s, but we are given +m, +s|+m, and +s|-m.
    - Gemini can we find this probability of +m|+s? We don't have the probability of +s, but we are given +m, +s|+m, and +s|-m.
      - Gemini response:
        Yes, using Bayes' Theorem, we can find the probability of +m|+s. The formula is $P(+m|+s) = \frac{P(+s|+m)P(+m)}{P(+s)}$. To calculate this, we would need the values for $P(+s|+m)$, $P(+m)$, and $P(+s)$.

        To find $P(+s)$, we can use the law of total probability, since we have $P(+s|+m)$ and $P(+s|-m)$ and we can assume that either +m or -m must be true (i.e., $P(+m) + P(-m) = 1$). The formula would be: $P(+s) = P(+s|+m)P(+m) + P(+s|-m)P(-m)$.

        So, if you provide the values for $P(+m)$, $P(+s|+m)$, and $P(+s|-m)$, we can calculate $P(+m|+s)$ for you.
      - Yes, we can do this by marginalization.

## Probabilistic models
- May not account for every variable
  - May not account for all interactions between vars
- They can erason about unknown vars, given evidence
  * They can reason about unknown variables, given evidence.
- Inferencing
- Independent
  - Variables are independent if the joint proobability distribution is a product of each of their distributions
  - P(x|y) = p(x)
    - P(x) will simply be note ffected by any evidence given by y
- In reality, very few variables are truly independent.
- Independence is a simplifying modeling assumption.
- You can test for independence by creating your own joint probability table to see if it matches the given probabilities.
    - This is probably relevant for test questions.

## Conditional Independence
  - Example: P(toothache, Cavity, Catch)
    - If there is a cavity, the probability of a catch does not depend on whether there is a toothache:
      P(+catch | +tootache, +cavity) = P(+catch | +cavity)
    - The same holds if there is no cavity:
      P(+catch | +tootache, -cavity) = P(+catch | -cavity)
    - Therefore, toothache is essentially irrelevant for predicting a catch.
    - We can state that Catching is conditionally independent of Toothache given Cavity:
      P(Catch | Tootache, Cavity) = P(Catch | cavity)
    - Conditional independence is the most basic and robust form of knowledge about *uncertain environments*.
    - X is conditionally independent of Y given Z is written as X || Y | Z.
  * They can reason about unknown variables, given evidence.


## Cond independence + chain rule
- Trivial Decomp:
  - P(T,R,U) = P R * P(T | R) * P(U|R, T)
- Cond indep can help us on chain rule! Simplify the decomp given independence
Jopint is way too big to represent with a lot of variables
Hard to learn empirically about more than few vars at a time
- Bayesian netowrks implicitly encode join distributions
- Why are we guaranteed that setting the join prob that way will result in a propr join distribution where it sums to 1
