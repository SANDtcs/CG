import turtle

iterations = 3

axiom = "F-G-G"
rule_set = {
    "F" : "F-G+F+G-F",
    "G" : "GG"
}
angle = 120
distance = 50


def applyRules(var, rule_set):
    return var if (x := rule_set.get(var, None)) is None else x


def processString(oldRule, rule_set):
    newstr = ""
    for ch in oldRule:
        newstr = newstr + applyRules(ch, rule_set)
    return newstr


def createLSystem(numIters, axiom, rule_set):
    startString = axiom
    endString = ""
    for i in range(numIters):
        endString = processString(startString, rule_set)
        startString = endString
    return endString


def drawLsystem(aTurtle, instructions, angle, distance):
    for cmd in instructions:
        if cmd == 'F'or cmd == "G":
            aTurtle.forward(distance)
        elif cmd == 'B':
            aTurtle.backward(distance)
        elif cmd == '+':
            aTurtle.right(angle)
        elif cmd == '-':
            aTurtle.left(angle)


inst = createLSystem(iterations, axiom, rule_set)
print(inst)

t = turtle.Turtle()
wn = turtle.Screen()

t.ht()
t.up()
t.back(200)
t.down()
t.speed(9)
drawLsystem(t, inst, angle, distance)
wn.exitonclick()
