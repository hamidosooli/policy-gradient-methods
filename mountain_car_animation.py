import numpy as np
import pygame as pg
import time


WIDTH = 800
HEIGHT = 800
AGENT_WIDTH = 50
AGENT_HEIGHT = 50
goal_x = 0.5
goal_y = 0.45 * np.sin(3 * goal_x) + 0.55


def draw_curve(scr):
    line_color = pg.Color(0, 0, 0)
    x = np.arange(-1.2, 0.5, 0.01)
    y = 0.45 * np.sin(3 * x) + 0.55
    points = []
    for i, j in zip(x, y):
        points.append([(i + 1.2) * (.6*WIDTH), HEIGHT // 1.6 - j * (HEIGHT // 2.45)])
    pg.draw.lines(scr, line_color, False, points, 5)


def animate(trajectory):
    bg_color = pg.Color(255, 255, 255)
    goal_color = pg.Color(0, 255, 0)
    agent_color = pg.Color(255, 0, 0)
    pg.init()  # initialize pygame
    pg.font.init()
    myfont = pg.font.SysFont('calibri', 35)
    # textsurface = myfont.render('REINFORCE', False, (0, 0, 255))
    # textsurface = myfont.render('REINFORCE with Baseline', False, (0, 0, 255))
    # textsurface = myfont.render('One-step Actor Critic', False, (0, 0, 255))
    textsurface = myfont.render('Actor Critic with Eligibility Traces', False, (0, 0, 255))
    screen = pg.display.set_mode((WIDTH+2, HEIGHT+2))   # set up the screen
    pg.display.set_caption("Hamid Osooli")              # add a caption
    bg = pg.Surface(screen.get_size())                  # get a background surface
    bg = bg.convert()
    bg.fill(bg_color)

    agent = pg.Surface((AGENT_WIDTH, AGENT_HEIGHT))
    agent.fill(agent_color)

    screen.blit(bg, (0,0))
    clock = pg.time.Clock()
    pg.display.flip()
    run = True
    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
        draw_curve(screen)
        for traj in trajectory:
            x = traj
            y = 0.45 * np.sin(3 * x) + 0.55
            print('x=', x, 'y=', y)

            draw_curve(screen)

            # draw goal
            goal_rect = pg.Rect((goal_x * (1.875*WIDTH), HEIGHT // 6.55 + goal_y * (HEIGHT // 25)),
                                (AGENT_WIDTH, AGENT_HEIGHT))
            pg.draw.rect(screen, goal_color, goal_rect)

            # agent
            screen.blit(agent,
                        ((x + 1.2) * (.55*WIDTH), HEIGHT // 1.69 - y * (HEIGHT // 2.5)))

            pg.display.flip()
            pg.display.update()

            # clean agent's footprint
            screen.blit(bg,
                        ((x + 1.2) * (.55*WIDTH), HEIGHT // 1.69 - y * (HEIGHT // 2.5)))
            # screen.blit(textsurface, (0, 0))
            time.sleep(.007)  # wait between the shows
        run = False
    pg.quit()