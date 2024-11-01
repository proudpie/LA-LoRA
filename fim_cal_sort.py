import json
import numpy as np

# Store file names and gradient values for subsequent visualization
pathA = '' 
pathB = '' 
num = ... # your number of samples

def load_and_calculate_gradient_averages(filename="gradients.json"):
    with open(filename, "r") as f:
        cumulative_gradients = json.load(f)
    gradient_averages = {}
    for name, grad_sum in cumulative_gradients.items():
        if isinstance(grad_sum, list):
            avg_value = np.mean(grad_sum)
        elif isinstance(grad_sum, np.ndarray):
            avg_value = np.mean(grad_sum)
        else:
            raise ValueError(f"Unsupported type for gradient: {type(grad_sum)}")
        gradient_averages[name] = avg_value
    return gradient_averages

def process_file(input_file, output_file):
    with open(input_file, 'r') as f, open(output_file, 'w') as out:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            layer_name = lines[i].strip()
            grad_a = float(lines[i + 1].strip())
            grad_b = float(lines[i + 3].strip())
            importance = (grad_a + grad_b) / 2
            out.write(f"{layer_name}: {importance}\n")

gradient_avg = load_and_calculate_gradient_averages("gradients.json")
with open(pathA, 'w') as f:
    for name, avg in gradient_avg.items()
        f.write(f"{n}\n")
        f.write(f"{avg / num}\n")   
process_file(pathA, pathB)
with open(pathB, 'r') as f:
    lines = f.readlines()
lines.sort(key = lambda x:float(x.split(': ')[1]),reverse = True)
with open(pathB, 'w') as f:
    f.writelines(lines)
with open(pathB, 'r') as f:
    important_layers = [line.strip().split(':')[0] for line in f.readlines()]
top_layers_A = important_layers[:38]
top_layers_B = [name[:-1] + 'B' if name.endwith('A') else name for name in top_layers_A]
top_layers = top_layers_A + top_layers_B

    
