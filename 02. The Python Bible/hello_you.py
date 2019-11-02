# Ask user for name?
name = input('What is your name?: ')

# Ask user for age?
age = input('What is your age?: ')

# Ask user for city?
city = input('Where do you live?: ')

# Ask user what they enjoy?

hoby = input('What do you do for relaxing?: ')

# Create output text
text = 'Your name is {} and you are {} years old. You live in {} and you love {}'
output = text.format(name, age, city, hoby)
# Print output to screen
print(output )



name = 'Goran'

age = 35

place = 'Bonn'

output = output = 'Hello my name is {} and I am {} year old. I live in {} and I love Python!'
output = output.format(name, age, place)

print(output)
