#include <windows.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <stdlib.h>
#include <bits/stdc++.h>

using namespace std;

class Point
{
public:
    int X;
    int Y;

    Point()
    {
        X=0;
        Y=0;
    }
    Point(int x,int y)
    {
        this->X=x;
        this->Y=y;
    }

    void xy(int x,int y)
    {
        this->X=x;
        this->Y=y;
    }
    int x()
    {
        return X;
    }
    int y()
    {
        return Y;
    }

};



void init(){
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0,640,0,480);
    glClearColor(0,0,0,0);
}


static void display(void)
{
    glColor3d(0,0,0);
    glFlush();
}


void plot(int x,int y){
    cout<<x<<" "<<y<<endl;
    glPointSize(1);
    glColor3d(1,1,1);
    glBegin(GL_POINTS);
    glVertex2d(x,y);
    glEnd();
    glFlush();
}


void beziercurve(vector<Point> pt){
    float t=0;
    int x,y;
    while(t<=1){

        x=pow(1-t,3)*pt[0].x()+3*t*pow(1-t,2)*pt[1].x()+3*pow(t,2)*(1-t)*pt[2].x()+pow(t,3)*pt[3].x();
        y=pow(1-t,3)*pt[0].y()+3*t*pow(1-t,2)*pt[1].y()+3*pow(t,2)*(1-t)*pt[2].y()+pow(t,3)*pt[3].y();
        plot(x,y);
        t+=0.001;
    }
}


vector<Point> point;
static void mouse(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        Point p(x,480-y);
        point.push_back(p);
    }
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN){
        beziercurve(point);
        point.erase(point.begin(),point.end());
    }
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitWindowSize(640,480);
    glutInitWindowPosition(10,10);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE );
    glutCreateWindow("Bezier Curve");
    init();
    glutDisplayFunc(display);
    glutMouseFunc(mouse);

    glutMainLoop();

    return EXIT_SUCCESS;
}
