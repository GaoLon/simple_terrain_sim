#!/usr/bin/python3

import atexit
import os
import signal
from threading import Lock
from tkinter import Frame, Label, Tk
import time
import rospy
from kimatic_simulator.msg import DiffCMD

UP = "w"
LEFT = "a"
DOWN = "s"
RIGHT = "d"
QUIT = "q"
AUTO = "g"

state = [False, False, False, False]
state_lock = Lock()
state_pub = None
root = None

vel = 0
omega = 0
auto = False

def keyeq(e, c):
    return e.char == c or e.keysym == c

def keyup(e):
    global state

    with state_lock:
        if keyeq(e, UP):
            state[0] = False
        elif keyeq(e, LEFT):
            state[1] = False
        elif keyeq(e, DOWN):
            state[2] = False
        elif keyeq(e, RIGHT):
            state[3] = False


def keydown(e):
    global state
    global auto

    with state_lock:
        if keyeq(e, QUIT):
            shutdown()
        elif keyeq(e, UP):
            state[0] = True
        elif keyeq(e, LEFT):
            state[1] = True
        elif keyeq(e, DOWN):
            state[2] = True
        elif keyeq(e, RIGHT):
            state[3] = True
        elif keyeq(e, AUTO):
            auto = not auto
            if auto:
                print("\033[92mSet mode to AUTO. \033[93m\U000026A0\033[0m")
            else:
                print("\033[92mSet mode to MANUAL. \033[93m\U000026A0\033[0m")

def publish_cb(_):
    global vel
    global omega
    global auto

    if auto:
        return
    with state_lock:
        if state[0]:
            vel = min(vel+0.1, max_velocity)
        elif state[2]:
            vel = max(vel-0.1, -max_velocity/2)
        else:
            vel = 0
        if state[1]:
            omega = min(omega+0.1, max_w)
        elif state[3]:
            omega = max(omega-0.1, -max_w)
        else:
            omega = 0

        command.velocity = vel
        command.angle_velocity = omega
        if state_pub is not None:
            state_pub.publish(command)


def exit_func():
    os.system("xset r on")


def shutdown():
    root.destroy()
    rospy.signal_shutdown("shutdown")


def main():
    global state_pub
    global root
    global command
    global max_velocity
    global max_w

    # max_velocity = 1.0
    # max_w = 2.0
    max_velocity = 10.0
    max_w = 3.0

    state_pub = rospy.Publisher(
        "/cmd", DiffCMD, queue_size=1
    )

    command = DiffCMD()

    rospy.Timer(rospy.Duration(0.05), publish_cb)

    atexit.register(exit_func)
    os.system("xset r off")

    root = Tk()
    frame = Frame(root, width=100, height=100)
    frame.bind("<KeyPress>", keydown)
    frame.bind("<KeyRelease>", keyup)
    frame.pack()
    frame.focus_set()
    lab = Label(
        frame,
        height=10,
        width=30,
        text="Focus on this window\nand use the WASD keys\nto drive the car.\n\nPress G to change mode\n\nPress Q to quit",
    )
    lab.pack()
    print("Press %c to quit" % QUIT)
    print("Press %c to change mode" % AUTO)
    print("\033[92mSet mode to MANUAL. \033[93m\U000026A0\033[0m")

    root.mainloop()


if __name__ == "__main__":
    rospy.init_node("keyboard_control", disable_signals=True)

    signal.signal(signal.SIGINT, lambda s, f: shutdown())
    time.sleep(1) 
    main()
