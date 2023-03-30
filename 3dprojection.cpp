#include <windows.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
GLfloat ex = 0.0, ey = -4.0, ez = -3.0;
void init()
{
        glEnable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        glFrustum(-4.0, 4.0, -4.0, 4.0, 1.0, 10.0);
        //glOrtho(-4.0, 4.0, -4.0,  4.0, 1.0, 10.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        gluLookAt(ex, ey, ez,  0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}
GLfloat vertices[8][3]={
    {-1.0, -1.0, 1.0},     {-1.0, 1.0, 1.0},    {1.0, 1.0, 1.0},
    {1.0, -1.0, 1.0},    {-1.0, -1.0, -1.0},     {-1.0, 1.0, -1.0},
    {1.0, 1.0, -1.0},    {1.0, -1.0, -1.0}
};

GLfloat vertices2[5][3]={
    {0.0, 2.0, 0.0},{-2.0,-2.0,-2.0},{2.0,-2.0,-2.0},{2.0,-2.0,2.0},{-2.0,-2.0,2.0}
};

GLfloat colors[8][3]={
    {0.0, 0.0, 0.0},{1.0, 0.0, 0.0},
    {1.0, 1.0, 0.0},{0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},{1.0, 0.0, 1.0},
    {1.0, 1.0, 1.0},{0.0, 1.0, 1.0}
};

void quad(int a, int b, int c, int d)
{
    glBegin(GL_QUADS);
        glVertex3fv(vertices2[a]);
        glVertex3fv(vertices2[b]);
        glVertex3fv(vertices2[c]);
        glVertex3fv(vertices2[d]);
    glEnd();
}

void tri(int a,int b,int c){
    glBegin(GL_TRIANGLES);
        glVertex3fv(vertices2[a]);
        glVertex3fv(vertices2[b]);
        glVertex3fv(vertices2[c]);
    glEnd();
}

void mydraw()
{
    glColor3f(1.0, 0.0, 0.0);
    glColor3fv(colors[0]);     quad(0, 3, 2, 1);
    glColor3fv(colors[1]);     quad(2, 3, 7, 6);
    glColor3fv(colors[2]);     quad(0, 4, 7, 3);
    glColor3fv(colors[3]);     quad(1, 2, 6, 5);
    glColor3fv(colors[4]);     quad(4, 5, 6, 7);
    glColor3fv(colors[5]);     quad(0, 1, 5, 4);
}


void drawPyramid(){
    //glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    glColor3f(1.0,0.0,0.0);
    glColor3fv(colors[5]);     quad(1,2,3,4);
    glColor3fv(colors[6]);     tri(0,1,2);
    glColor3fv(colors[7]);     tri(0,2,3);
    glColor3fv(colors[3]);     tri(0,3,4);
    glColor3fv(colors[4]);     tri(0,1,4);

}


void drawObject()
{
    //glColor3f(0.7, 0.0, 0.50);
    glColor3f(0.0, 1.0, 1.0);
    //glutWireDodecahedron();

   //glutSolidTorus(0.5, 1, 4, 20);
   //glutWireTorus(3, 5, 25, 25);
   //glutWireSphere(2, 16, 16);

   //glTranslatef(-2.0, 1.0, 0);
   //glutWireTorus(0.5, 1, 30, 30);
   //glutWireTeapot(2.0);
  //glScalef(3.5, 3.5, 3.50);
   //glutWireCube(2.0);
   //glutWireTetrahedron();

  //glutWireOctahedron();
    //glBegin();
    //glEnd();

}
void resize(int width, int height)
{
    double aspect;
    glViewport(0, 0, width, height);
    aspect = (double) width / (double) height;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    cout << "aspect ratio" << aspect;
    if (aspect < 1.0)
    {
        glOrtho(-4., 4., -4./aspect, 4./aspect, 1., 100.);
    }
    else
    {
        glOrtho(-4.*aspect, 4.*aspect, -4., 4., 1., 100.);
    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0., 0., 5., 0., 0., 0., 0., 1., 0.);

}

void display(void)
{
    glClearColor(0.0f, 0.0f, 0.0, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //mydraw();
    //drawObject();
    drawPyramid();
    glutSwapBuffers();
}


static void key(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27 :
        case 'q':
            exit(0);
            break;

        case '+':
            cout << "+" << "Pressed ";
            ez += 1.0;
            break;

        case '-':
            cout << "+" << "Pressed ";
            ez -= 1.0;
            break;
    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(ex, ey, ez,  0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    glutPostRedisplay();

}


static void arrowKey(int key, int x, int y)
{
    cout << key << "pressed\n";
    switch(key)
    {
        case GLUT_KEY_LEFT:
            cout << "GLUT_KEY_LEFT" << "Pressed ";
            ex -= 1.0;
            break;

        case GLUT_KEY_RIGHT:
            cout << "GLUT_KEY_RIGHT" << "Pressed ";
            ex += 1.0;
            break;

        case GLUT_KEY_UP:
            cout << "GLUT_KEY_UP" << "Pressed ";
            ey -= 1.0;
            break;

       case GLUT_KEY_DOWN:
            cout << "GLUT_KEY_DOWN" << "Pressed ";
            ey += 1.0;
            break;

    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(ex, ey, ez,  0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    glutPostRedisplay();
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
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

    glutCreateWindow("3D Projection");

    glutReshapeFunc(resize);
    glutDisplayFunc(display);
    glutKeyboardFunc(key);
    glutSpecialFunc(arrowKey);
    //glutIdleFunc(idle);

    init();

    glutMainLoop();

    return EXIT_SUCCESS;
}
