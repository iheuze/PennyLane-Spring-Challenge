import json
import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit(weights):
    for i in range(len(weights)):
        qml.RX(weights[i, 0], wires=0)
        qml.RY(weights[i, 1], wires=1)
        qml.RZ(weights[i, 2], wires=2)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])
        
    # Return the Y0xZ2 observable
    return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))

def parameter_shift(weights):
    """Compute the gradient of the variational circuit given by the
    circuit function using the parameter-shift rule.

    Args:
        weights (array): An array of floating-point numbers with size (2, 3).

    Returns:
        array: The gradient of the variational circuit. The shape should match
        the input weights array.
    """

    # Return the gradient calculated using the parameter-shift rule
    gradient = np.zeros_like(weights)
    qnode = qml.QNode(circuit, dev)

    for i in range(len(weights)):
        for j in range(len(weights)):
            copy_weight = weights.copy()
            copy_weight[i][j] += np.pi/2
            forward_shift = qnode(copy_weight)  

            copy_weight[i][j] -= np.pi
            backward_shift = qnode(copy_weight) 

            gradient[i][j] = 0.5 * (forward_shift - backward_shift)
    return gradient



# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    
    ins = np.array(json.loads(test_case_input), requires_grad = True)
    out = parameter_shift(ins).round(3).tolist()
    
    return str(out)
    
def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    
    assert np.allclose(solution_output, expected_output, atol=1e-2), "The gradient calculated isn't quite right."


# These are the public test cases
test_cases = [
    ('[[1,0.5,-0.765], [0.1,0,-0.654]]', '[[0.0, 0.0, 0.0], [0.0, -0.455, 0.0]]'),
    ('[[0.94,-0.2,6.03],[-2.6,-0.058,1.2]]', '[[0.03, -0.039, 0.0], [-0.034, 0.166, 0.0]]')
]

# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
