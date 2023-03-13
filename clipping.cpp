#include <windows.h>
#include <GL/glut.h>
#include <bits/stdc++.h>
using namespace std;

#define MAXWIDTH 640
#define MAXHEIGHT 480

class Point
{
    public:
        int x;
        int y;

        Point(int xp=0, int yp=0)
        {
            x=xp;
            y=yp;
        }
};
vector<Point> p;
vector<vector<Point>> points;
int c1,c2;
bool flag;
int xmin,ymin,xmax,ymax;


void init()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0,MAXWIDTH,0,MAXHEIGHT);

}

static void display(void)
{
    glColor3d(0,0,0);
    glFlush();
}

void plot(Point point)
{
    glColor3d(1,1,1);
    glPointSize(2);
    glBegin(GL_POINTS);
    glVertex2d(point.x, point.y);
    glEnd();
    glFlush();
}

void plotWindow(vector<Point> point)
{
    xmin=point[0].x;
    xmax=point[1].x;
    ymin=point[0].y;
    ymax=point[1].y;

    glColor3d(1,1,1);
    glLineWidth(3);
    glBegin(GL_LINE_LOOP);
    glVertex2d(point[0].x, point[0].y);
    glVertex2d(point[0].x, point[1].y);
    glVertex2d(point[1].x, point[1].y);
    glVertex2d(point[1].x, point[0].y);
    glEnd();
    glFlush();
}

void plotPolygon(vector<Point> point, int r, int g, int b)
{
    glColor3d(r,g,b);
    glLineWidth(2);
    glBegin(GL_LINE_LOOP);
    for(int i=0;i<point.size();i++)
    {
        glVertex2d(point[i].x, point[i].y);
    }
    glEnd();
    glFlush();
}
void print(vector<Point> point)
{
    for(int i=0;i<point.size();i++)
    {
        cout<< point[i].x<< " "<<point[i].y<<endl;
    }
    cout<<endl;
}

vector<Point> clipLeft(vector<Point> point)
{
    vector<Point> result;
    int n=point.size();

    for(int i=0,j=1;i<n;i++,j=(++j)%n)
    {
        float x1=point[i].x,y1=point[i].y,x2=point[j].x,y2=point[j].y;
        if(x1>xmin && x2 > xmin)
        {
            result.push_back(point[j]);
        }

        else if(x1<xmin && x2>xmin)
        {
            int x=xmin;
            float m = (y2-y1)/(x2-x1);
            float y = y1+(x-x1)*m;
            Point intersection=Point(x,y);
            result.push_back(intersection);
            result.push_back(point[j]);
        }

        else if(x1>xmin && x2<xmin)
        {
            int x=xmin;
            float m = (y2-y1)/(x2-x1);
            float y=y1+(x-x1)*m;
            Point intersection=Point(x,y);
            result.push_back(intersection);
        }
    }
    return result;
}

vector<Point> clipBottom(vector<Point> point)
{
    vector<Point> clipped;
    int n=point.size();

    for(int i=0,j=1;i<point.size();i++,j=(++j)%n)
    {
        float x1=point[i].x,y1=point[i].y,x2=point[j].x,y2=point[j].y;

         if(y1<ymin && y2>=ymin)
        {
            int y=ymin;
            float m=(y-y1)/(y2-y1);
            int x=x1+(x2-x1)*m;
            Point intersection=Point(x, y);
            clipped.push_back(intersection);
            clipped.push_back(point[j]);
        }
        // both inside
        else if(y1>=ymin && y2>=ymin)
        {
            clipped.push_back(point[j]);
        }
        // first inside, second outside
        else if(y1>=ymin && y2<ymin)
        {
            int y=ymin;
            float m=(y-y1)/(y2-y1);
            int x=x1+(x2-x1)*m;
            Point intersection=Point(x, y);
            clipped.push_back(intersection);
        }
    }
    return clipped;
}

vector<Point> clipRight(vector<Point> point)
{
    vector<Point> clipped;
    int n=point.size();

    for(int i=0,j=1;i<n;i++,j=(++j)%n)
    {
        float x1=point[i].x,y1=point[i].y,x2=point[j].x,y2=point[j].y;
        if(x1>xmax && x2<=xmax)
        {
            int x=xmax;
            float m=(y2-y1)/(x2-x1);
            int y=y1+(x-x1)*m;
            Point intersection=Point(x, y);
            clipped.push_back(intersection);
            clipped.push_back(point[j]);
        }
        // both inside
        else if(x1<=xmax && x2<=xmax)
        {
            clipped.push_back(point[j]);
        }
        // first inside, second outside
        else if(x1<=xmax && x2>xmax)
        {
            int x=xmax;
            float m=(y2-y1)/(x2-x1);
            int y=y1+(x-x1)*m;
            Point intersection=Point(x, y);
            clipped.push_back(intersection);
        }
    }
    return clipped;
}

vector<Point> clipTop(vector<Point> point)
{
    vector<Point> clipped;
    int n=point.size();

    for(int i=0,j=1;i<n;i++,j=(++j)%n)
    {
        float x1=point[i].x,y1=point[i].y,x2=point[j].x,y2=point[j].y;

        if(y1<ymax && y2<ymax)
        {
            clipped.push_back(point[j]);
        }

        else if(y1>ymax && y2<ymax)
        {
            int y=ymax;
            float m = (y-y1)/(y2-y1);
            int x = x1+m*(x2-x1);
            Point intersection = Point(x,y);
            clipped.push_back(intersection);
            clipped.push_back(point[j]);
        }
        else if(y1<ymax && y2>ymax)
        {
            int y=ymax;
            float m = (y-y1)/(y2-y1);
            int x = x1+m*(x2-x1);
            Point intersection = Point(x,y);
            clipped.push_back(intersection);
        }
    }
    return clipped;
}


void sutherlandHodgman(vector<Point> point)
{
    vector<Point> clippedPolygon;
    clippedPolygon=clipLeft(point);
    //plotPolygon(clippedPolygon,1,0,0);
    clippedPolygon=clipBottom(clippedPolygon);
    //print(clippedPolygon);
    clippedPolygon=clipRight(clippedPolygon);
    clippedPolygon=clipTop(clippedPolygon);
    plotPolygon(clippedPolygon,1,0,0);
}

static void mouse(int button, int state, int x, int y)
{
    y = MAXHEIGHT-y;
    if(button==GLUT_LEFT_BUTTON && state==GLUT_DOWN)
    {
        Point temp(x,y);
        plot(temp);
        p.push_back(temp);
        if(!flag)
        {
            c2++;
            if(c2==2)
            {
                flag=true;
                c2=1;
                c1=1;
                points.push_back(p);
                p.erase(p.begin(),p.end());
                plotWindow(points[0]);
            }
        }
    }

    else if(button==GLUT_RIGHT_BUTTON && state==GLUT_DOWN)
    {
        points.push_back(p);
        p.erase(p.begin(),p.end());
        for(;c1<points.size();c1++)
        {
            plotPolygon(points[c1],1,1,1);
        }
    }

}


static void key(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'c':
            while(c2<c1)
                sutherlandHodgman(points[c2++]);
    }

    glutPostRedisplay();
}


int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitWindowSize(MAXWIDTH,MAXHEIGHT);
    glutInitWindowPosition(10,10);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutCreateWindow("Polygon Clipping");

    init();
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutKeyboardFunc(key);

    glutMainLoop();

    return EXIT_SUCCESS;
}
