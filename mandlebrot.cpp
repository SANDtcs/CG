#include <windows.h>
#include <GL/glut.h>
#include <bits/stdc++.h>
using namespace std;

void init()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0,640,0,480);
    glClearColor(0,0,0,0);
}

void plot(int x, int y, int color)
{
    glColor3f(color,color,color);
    glPointSize(1);
    glBegin(GL_POINTS);
    glVertex2i(x, y);
    glEnd();
    glFlush();
}

void mandelBrot()
{
    float Re_min=-2, Re_max=0.5, Im_min=-1.25, Im_max=1.25;
    int N=64;

    complex<float> z;

    int width = 640, height=480;

    int i,j, count,color;
    float delta = (Re_max - Re_min)/width;
    for (i = 0, z.real(Re_min); i < width; i++, z.real(z.real()+ delta))
    {
        for (j = 0, z.imag(Im_min); j < height; j++, z.imag(z.imag()+ delta))
        {
            count = 0;
            complex<float> x(0,0);
            while (abs(x) <= 2.0 && count < N)
            {
                x = x*x+z;
                count++;
            }
            if (abs(x) <= 2.0)
                color=0;
            else
                color = int(1 + 254* count/(float)N);
            plot(i,j,color);
        }
    }
}

void julia()
{
    float Re_min=-1.2, Re_max=1.2, Im_min=-1.2, Im_max=1.2;
    int N=128;
    complex<float> z(-0.74543,0.11301);
    int width = 640, height=480;
    int i,j, count,color;
    float delta = (Re_max - Re_min)/width;
    complex<float> x;
    for (i = 0, x.real(Re_min); i < width; i++, x.real(x.real()+ delta))
    {
        for (j = 0, x.imag(Im_min); j < height; j++, x.imag(x.imag()+ delta))
        {
            count = 0;
            complex<float> y = x;
            while (abs(y) <= 2.0 && count < N)
            {
                y = y*y+z;
                count++;
            }
            if (abs(y) <= 2.0)
                color=0;
            else
                color = int(1 + 254* count/(float)N);
            plot(i,j,color);
        }
    }

}

static void display(void)
{
    //glClear(GL_COLOR_BUFFER_BIT);
    glColor3d(0,0,0);
    glFlush();
    /*int choice;
    cout<<"1.MandelBrot Set\n2.Julia \n Enter your choice:";
    cin>>choice;
    switch(choice)
    {
        case 1:
            mandelBrot();
            break;
        case 2:
            julia();
            break;
    }*/

     //mandelBrot();
     julia();
}

static void idle(void)
{
    glutPostRedisplay();
}


int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitWindowSize(640,480);
    glutInitWindowPosition(10,10);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutCreateWindow("Fractals");
    init();
    glutDisplayFunc(display);
    glutMainLoop();

    return EXIT_SUCCESS;
}

