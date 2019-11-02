word = "antidisestablishmentarianism"

# use a slice to take out the word "establishment"
# and store it in the answer variable....
print(word.index('establishment'))
print(word.index('arianism'))
answer = word[word.index('establishment'):word.index('arianism')]
print(answer)
