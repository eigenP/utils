import re

with open('src/eigenp_utils/single_cell.py', 'r') as f:
    content = f.read()

# Replace the block where we map .obs to incorporate mapping_confidence correctly.
pattern = r"        for i in range\(N_query\):\n\s*idx = pruned_indices\[i\]\n\s*dist = pruned_distances\[i\]\n\s*labels_i = ref_labels\[idx\]\n\s*# Inverse distance weighting for votes\n\s*weights = 1\.0 / \(dist \+ 1e-8\)\n\s*weights /= np\.sum\(weights\)\n\s*# Tally weighted votes\n\s*vote_tally = \{\}\n\s*for lbl, w in zip\(labels_i, weights\):\n\s*vote_tally\[lbl\] = vote_tally\.get\(lbl, 0\) \+ w\n\s*# Find the winner\n\s*winner = max\(vote_tally\.items\(\), key=lambda x: x\[1\]\)\n\s*winning_label = winner\[0\]\n\s*winning_weight_frac = winner\[1\] # Already normalized to 1\.0\n\s*mapped_labels\.append\(winning_label\)\n\s*# Mapping confidence: proportion of weighted vote\n\s*# Optionally penalized by the average distance to neighbors\n\s*# Let's start with just the proportion of the weighted vote\n\s*confidences\.append\(winning_weight_frac\)"

replacement = """        for i in range(N_query):
            idx = pruned_indices[i]
            dist = pruned_distances[i]

            labels_i = ref_labels[idx]

            # Inverse distance weighting for votes
            weights = 1.0 / (dist + 1e-8)
            weights /= np.sum(weights)

            # Tally weighted votes
            vote_tally = {}
            for lbl, w in zip(labels_i, weights):
                vote_tally[lbl] = vote_tally.get(lbl, 0) + w

            # Find the winner
            winner = max(vote_tally.items(), key=lambda x: x[1])
            winning_label = winner[0]
            winning_weight_frac = winner[1] # Already normalized to 1.0

            mapped_labels.append(winning_label)

            # Mapping confidence: proportion of weighted vote
            confidences.append(winning_weight_frac)"""

content = re.sub(pattern, replacement, content)

with open('src/eigenp_utils/single_cell.py', 'w') as f:
    f.write(content)
