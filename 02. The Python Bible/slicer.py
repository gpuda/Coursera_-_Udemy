# get use email address
email = input('What is your email adress? ').strip()

# slice our user name
user = email[:email.index('@')]

# slice domain name
domain = email[email.index('@')+1:]


# format message
message = 'Your email is: {}, your username is: {} \
and your domain is: {}'.format(email, user, domain)

# display output message
print(message)
