import requests
import json

url = "http://127.0.0.1:1234/v1/chat/completions"
model = "deepseek/deepseek-r1-0528-qwen3-8b"

system_prompt = ("You are a helpful assistant. Solve this puzzle for me. You can represent actors with a_1, a_2, ... "
                 "and agents with A_1, A_2, ... . Your solution must be a list of boat moves where each move indicates "
                 "the people on the boat. For example, if there were two actors and two agents, you should return: "
                 "moves=[[\"A_2\", \"a_2\"], [\"A_2\"], [\"A_1\", \"A_2\"], [\"A_1\"], [\"A_1\", \"a_1 \"]] which indicates "
                 "that in the first move, A_2 and a_2 row from left to right, and in the second move, A_2 rows from right "
                 "to left and so on. Requirements: • The list shouldn't have comments. • Ensure your "
                 "final answer also includes the complete list of moves for final solution. "
                 "IMPORTANT: Your response must have the correct format as this is vital for the evaluation.")

user_prompt = ("2 actors and their 2 agents want to cross a river in a boat that is capable of holding "
               "only 2 people at a time, with the constraint that no actor can be in the presence of another agent, "
               "including while riding the boat, unless their own agent is also present, because each agent is worried "
               "their rivals will poach their client. Initially, all actors and agents are on the left side of the river "
               "with the boat. How should they cross the river? (Note: the boat cannot travel empty)")

payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    "temperature": 0.0,
    "max_tokens": 2048,   # <<--- CLAVE: sube este valor si hace falta
}

resp = requests.post(url, json=payload, timeout=120)
resp.raise_for_status()
data = resp.json()

print("finish_reason:", data["choices"][0].get("finish_reason"))
print(data["choices"][0]["message"]["content"])
