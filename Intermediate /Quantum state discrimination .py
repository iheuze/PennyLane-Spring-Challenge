import json
import pennylane as qml
import pennylane.numpy as np

def maximal_probability(theta_1, theta_2, p_1, p_2):

    """
    This function calculates the maximal probability of distinguishing
    the states

    Args:
        theta_1 (float): Angle parametrizing the state |phi_1>.
        theta_2 (float): Angle parametrizing the state |phi_2>.
        p_1 (float): Probability that the state was |phi_1>.
        p_2 (float): Probability that the state was |phi_2>.

    Returns:
        (Union[float, np.tensor]): Maximal probability of distinguishing the states.
    
    """

    # defining the states
    phi_1 = np.array([np.cos(theta_1), np.sin(theta_1)])
    phi_2 = np.array([np.cos(theta_2), np.sin(theta_2)])

    # calculating q, the outer product of the states
    q_1 = np.outer(phi_1, phi_1)
    q_2 = np.outer(phi_2, phi_2)

    # finding the success probability
    p_succ = p_1 * q_1 - p_2 * q_2

    # using eigenvalues to maximise p_succ 
    eigenvals = np.linalg.eigvals(p_succ)

    # Return the highest probability of distinguishing the states
    return 0.5 + 0.5 * np.sum(np.abs(eigenvals))

# These functions are responsible for testing the solution.


def run(test_case_input: str) -> str:
    theta1, theta2, p_1, p_2 = json.loads(test_case_input)
    prob = np.array(maximal_probability(theta1, theta2, p_1, p_2)).numpy()

    return str(prob)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    assert np.allclose(solution_output, expected_output, rtol=1e-4)


# These are the public test cases
test_cases = [
    ('[0, 0.7853981633974483, 0.25, 0.75]', '0.8952847075210476'),
    ('[1.83259571459, 1.88495559215, 0.5, 0.5]', '0.52616798')
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
