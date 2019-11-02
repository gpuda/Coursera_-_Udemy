# get sentence from user

original = input("Please enter a sentence: ").strip().lower() #  => egg glove

# split sentence into words

words = original.split()  # => [egg, glove]

# loop through words and convert to pig latin

new_words = []

for word in words:
    if word[0] in "aeiou":
        new_word = word + "yay"    # eggyay
        new_words.append(new_word)
    else:
        vowel_pos = 0
        for letter in word:
            if letter not in "aeiou":
                vowel_pos = vowel_pos + 1
            else:
                break
        cons = word[:vowel_pos]  # gl
        the_res = word[vowel_pos:] # ove
        new_word = the_res + cons + "ay"  # oveglay
        new_words.append(new_word)

#RULES FOR PIG LATIN
# if is starts vowel, just add "yay"
# otherwise, move the first consonant cluster to end, and add "ay"

# stick words back together

output = " ".join(new_words)

# output the final string
print(output)
