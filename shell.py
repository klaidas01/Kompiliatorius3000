import LiT

while True:
    text = input()
    result, error = LiT.run('input file name', text)

    if error: print(error.toString())
    else: print(result)