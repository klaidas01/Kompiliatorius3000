import LiT

while True:
    text = input()
    if text.strip() == "": continue
    result, error = LiT.run('input file name', text)

    if error: print(error.toString())
    elif result:
       if len(result.elements) == 1:
        print(repr(result.elements[0]))
       else:
        print(repr(result))