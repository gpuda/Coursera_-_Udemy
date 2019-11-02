import random


difficulty = int(input("Your difficulty:"))
print(difficulty)

health = 50
potion_health = int(random.randint(25,50) / difficulty)

health = health + potion_health
print("Your health is:", health)
