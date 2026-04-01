import random
import os

output_file = "data/crypto_ai_dataset.txt"
target_size = 25 * 1024 * 1024  # 25MB

# Base components representing common human patterns
base_words = ["password", "qwerty", "admin", "login", "hello", "welcome", "iloveyou", "princess", "football", "baseball", "dragon", "starwars", "monkey", "sunshine", "shadow", "secret", "master", "spring", "summer", "autumn", "winter"]
years = [str(year) for year in range(1970, 2025)]
symbols = ["!", "@", "#", "$", "%", "123", "1234", "1"]

def generate_random_password():
    # Model 1: word + digits
    if random.random() < 0.4:
        return random.choice(base_words).capitalize() + random.choice(years)
    # Model 2: word + symbols
    elif random.random() < 0.7:
        word = random.choice(base_words)
        # Random leetspeak
        word = word.replace('e', '3').replace('a', '@').replace('o', '0').replace('s', '$')
        return word + random.choice(symbols)
    # Model 3: pure word
    elif random.random() < 0.8:
        return random.choice(base_words)
    # Model 4: word + word
    elif random.random() < 0.9:
        return random.choice(base_words) + random.choice(base_words)
    # Model 5: number sequence
    else:
        seqs = ["123456", "123456789", "111111", "123123"]
        return random.choice(seqs)

print(f"Generating synthetic structured dataset at {output_file}...")
bytes_written = 0
with open(output_file, "w", encoding="utf-8") as f:
    while bytes_written < target_size:
        pwd = generate_random_password() + "\n"
        f.write(pwd)
        bytes_written += len(pwd.encode('utf-8'))

print(f"Dataset generated. Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
