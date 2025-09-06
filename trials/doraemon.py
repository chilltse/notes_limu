import turtle as t

t.setup(600, 700)
t.speed(0)
t.hideturtle()
t.colormode(255)

# 画圆函数（填充 + 轮廓）
def draw_circle(x, y, radius, color_fill, color_outline='black', pensize=2):
    t.penup()
    t.goto(x, y - radius)
    t.pendown()
    t.pensize(pensize)
    t.color(color_outline, color_fill)
    t.begin_fill()
    t.circle(radius)
    t.end_fill()

# 画头
draw_circle(0, 120, 150, (0, 160, 222))

# 画脸
draw_circle(0, 100, 120, 'white')

# 画眼睛
draw_circle(-50, 180, 25, 'white')
draw_circle(50, 180, 25, 'white')
draw_circle(-50, 190, 10, 'black')
draw_circle(50, 190, 10, 'black')

# 画鼻子
draw_circle(0, 140, 18, 'red')
t.penup()
t.goto(0, 140)
t.pendown()
t.pensize(2)
t.goto(0, 110)

# 画嘴巴
t.penup()
t.goto(-90, 100)
t.pendown()
t.pensize(3)
t.setheading(-60)
t.circle(100, 120)

# 画胡须
t.pensize(2)
for y in [130, 110, 90]:
    t.penup()
    t.goto(-10, y)
    t.pendown()
    t.goto(-140, y - 20)
    t.penup()
    t.goto(10, y)
    t.pendown()
    t.goto(140, y - 20)

# 画身体
draw_circle(0, -80, 120, (0,160,222))

# 画领子
t.penup()
t.goto(-120, 40)
t.pendown()
t.pensize(6)
t.color('red')
t.goto(120, 40)

# 画铃铛
draw_circle(0, 20, 25, 'yellow')
t.penup()
t.goto(0, 20)
t.pendown()
t.pensize(3)
t.color('black')
t.dot(8)

t.penup()
t.goto(0, -5)
t.pendown()
t.goto(0, -30)

# 画口袋
t.penup()
t.goto(-70, -30)
t.pendown()
t.pensize(4)
t.color('black')
t.setheading(-60)
t.circle(70, 120)

# 画手臂
for angle in [120, 60]:
    t.penup()
    t.goto(0, -50)
    t.setheading(angle)
    t.pendown()
    t.pensize(10)
    t.forward(80)
    t.pensize(2)
    t.circle(10)

# 画腿
for x in [-40, 40]:
    t.penup()
    t.goto(x, -200)
    t.pendown()
    t.pensize(10)
    t.setheading(-90)
    t.forward(60)
    t.pensize(2)
    t.circle(15)

t.done()

