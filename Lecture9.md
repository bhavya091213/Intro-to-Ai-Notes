# Lecture 9 - Minimax

## Pruning
- Minimax pruning, speeding up
- Alpha beta pruning
- Using domain specific knowledge to compute approxiamte answer are evaluation functions
- Alpha beta pruning are general purpose, computing exact answer
- MIN version
  - We are computing the min value at some node N
  - we are looping over N's children
  - N's estimate of the childrens' min is dropping
  - Who cares about N's value? Max cares about it
  - We denote a to be the best value that MAX can get at any choce point along the current path from the root
  - if N becoems worse than A, max will avoid it, so we can stop cinsidering N's other children. (it is already bad enough that it won't be played)
  - Gemini write pseudocode for this and put it in markdown code block  
    - Gemini response:
      function alpha_beta_pruning_min(node, alpha, beta):
          if node is a leaf:
              return evaluate(node)

          value = infinity
          for child in node.children:
              value = min(value, alpha_beta_pruning_max(child, alpha, beta))
              if value <= alpha:
                  break  # Beta cutoff
              beta = min(beta, value)
          return value

      function alpha_beta_pruning_max(node, alpha, beta):
          if node is a leaf:
              return evaluate(node)

          value = -infinity
          for child in node.children:
              value = max(value, alpha_beta_pruning_min(child, alpha, beta))
              if value >= beta:
                  break  # Alpha cutoff
              alpha = max(alpha, value)
          return value
    - If you are the max player, how are you going to compare the value with beta?
    -