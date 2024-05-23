import random
import time
import numpy as np
from tqdm import tqdm

def generate_numbers():
  # Generate three random numbers between 0 and 15
  numbers = [random.randint(0, 15) for _ in range(3)]
  
  # Calculate the fourth number so that the sum is exactly 15
  total = sum(numbers)
  fourth_number = 15 - total
  
  # If the fourth number is out of range (0 to 15), start over
  if fourth_number < 0 or fourth_number > 15:
      return generate_numbers()
  
  numbers.append(fourth_number)
  random.shuffle(numbers)  # Shuffle to ensure random order
  return numbers


def generate_numbers2():
  # Generate three random points in the range [0, 15)
  points = sorted(random.sample(range(1, 15), 3))
  
  # Calculate the segments between points, plus the ends of the range
  num1 = points[0]
  num2 = points[1] - points[0]
  num3 = points[2] - points[1]
  num4 = 15 - points[2]
  
  return [num1, num2, num3, num4]




start_time = time.time()

numbers = []
for _ in tqdm(range(1000000)):
  numbers.append(generate_numbers())
numbers = np.array(numbers).T

sum = 0
for line in numbers.T :
  sum += np.sum(line)
sum /= 1000000
print(sum)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")



count = []
for facies in numbers:
  count.append([list(facies).count(i) for i in range(16)])

# Print the counts
for j in range(len(numbers)):
  print(f'\nfacies{j}')
  for i, c in enumerate(count[j]):
    print(f"Count of {i}: {c}")

# import matplotlib.pyplot as plt

# # Plot histogram bars
# plt.bar(range(16), count)
# plt.xlabel('Number')
# plt.ylabel('Count')
# plt.title('Number Frequency')
# plt.show()



import itertools
from collections import Counter

def generate_combinations():
    result = []
    # Iterate over all possible values for a, b, and c
    for a in np.arange(0,15+0.5,0.5):
        for b in np.arange(0,15+0.5,0.5):
            for c in np.arange(0,15+0.5,0.5):
                d = 15 - (a + b + c)
                if 0 <= d <= 15:
                    result.append((a, b, c, d))
    return result

# Generate all valid combinations
combinations = generate_combinations()

# Count occurrences of each number
counts = Counter()
for combination in combinations:
    counts.update(combination)

# Print combinations and their counts
print(f"Total combinations: {len(combinations)}")
print(f"Counts: {counts}")

# Print a few sample combinations
import random
sample_combinations = random.sample(combinations, 10)
print("Sample combinations:")
for combo in sample_combinations:
    print(combo)