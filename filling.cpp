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


class Color
{
    public:
        float r;
        float g;
        float b;

        Color(float rr=0, float gg=0, float bb=0)
        {
            r=rr;
            g=gg;
            b=bb;
        }

        Color(float pixel[3])
        {
            r=pixel[0];
            g=pixel[1];
            b=pixel[2];
        }
};

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
    glPointSize(1);
    glColor3d(1,1,1);
    glPointSize(1);
    glBegin(GL_POINTS);
    glVertex2d(point.x, point.y);
    glEnd();
    glFlush();
}


void plotPolygon(vector<Point> point,Color c)
{
    glColor3d(c.r,c.g,c.b);
    glLineWidth(2);
    glBegin(GL_LINE_LOOP);
    for(int i=0;i<point.size();i++)
    {
        glVertex2d(point[i].x, point[i].y);
    }
    glEnd();
    glFlush();
}

void plotPolygon(vector<Point> point,int r, int g, int b)
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

bool sameColor(Color c1, Color c2)
{
    return (fabs(c1.r-c2.r)<0.05 && fabs(c1.g-c2.g)<0.05 && fabs(c1.b-c2.b)<0.05);
}

void bFill(int x, int y, Color fillColor, Color boundary)
{
    float pixel[3];
    glReadPixels(x,y,1,1,GL_RGB,GL_FLOAT,&pixel);

    Color current(pixel);

    if(!sameColor(current,boundary) && !sameColor(current,fillColor))
    {
        glPointSize(1);
        glBegin(GL_POINTS);
        glColor3f(fillColor.r,fillColor.g,fillColor.b);
        glVertex2d(x,y);
        glEnd();
        glFlush();

    bFill(x+1,y,fillColor,boundary);
    bFill(x-1,y,fillColor,boundary);
    bFill(x,y+1,fillColor,boundary);
    bFill(x,y-1,fillColor,boundary);
    }

}

void boundaryFill(int button, int state, int x, int y)
{
    vector<Point> polygon;
    y=MAXHEIGHT-y;

    if(button==GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        Point temp = Point(x,y);
        plot(temp);
        p.push_back(temp);
    }
    else if(button==GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
    {
        Point interior(x,y);
        Color boundary(1.0f,1.0f,1.0f);
        Color fillColor(0.0f,1.0f,0.0f);

        plotPolygon(p,boundary);

        bFill(interior.x, interior.y,fillColor,boundary);
    }
}

void flood(int x, int y, Color fillColor, Color oldColor)
{
    float pixel[3];
    glReadPixels(x,y,1,1,GL_RGB,GL_FLOAT,&pixel);

    Color current(pixel);
    if(sameColor(current,oldColor))
    {
        glBegin(GL_POINTS);
        glColor3f(fillColor.r, fillColor.g, fillColor.b);
        glVertex2d(x,y);
        glEnd();
        glFlush();

        flood(x+1,y,fillColor,oldColor);
        flood(x-1,y,fillColor,oldColor);
        flood(x,y+1,fillColor,oldColor);
        flood(x,y-1,fillColor,oldColor);
    }

}

void floodFill(int button, int state, int x, int y)
{
    vector<Point> polygon;
    y=MAXHEIGHT-y;

    if(button==GLUT_LEFT_BUTTON && state==GLUT_DOWN)
    {
        Point temp = Point(x,y);
        plot(temp);
        p.push_back(temp);
    }
    else if(button==GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
    {
        Point interior(x,y);
        Color oldColor(0.0f,0.0f,0.0f);
        Color fillColor(0.0f,1.0f,0.0f);

        plotPolygon(p,fillColor);
        p.erase(p.begin(),p.end());
        flood(interior.x, interior.y, fillColor, oldColor);
    }
}

void yx(int button, int state, int xx, int yy)
{
   static vector<Point> polygon;
   yy=MAXHEIGHT-yy;

    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        plot(Point(xx, yy));

        Point point(xx, yy);
        polygon.push_back(point);

        glFlush();
    } else if(button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        plotPolygon(polygon, 1, 1, 1);
        glFlush();

        Color background(0.0f, 0.0f, 0.0f);

        for(int y = 0; y <= MAXHEIGHT; y++) {      // For each scan line
            vector<int> intersections;
            int n = polygon.size();

            for(int i = 0, j = 1; i < n; i++, j = (++j) % n) {      // For each edge
                /*
                if(polygon[i].x != polygon[j].x) {
                    int x = scanLineIntersect(y, polygon[i], polygon[j]);
                    intersections.push_back(x);
                }
                */

                int y1 = polygon[i].y;
                int y2 = polygon[j].y;

                if(y1 < y2) {
                    if(y >= y1 && y < y2) {
                        int x = (y - y1) * (polygon[j].x - polygon[i].x) / (y2 - y1) + polygon[i].x;
                        intersections.push_back(x);
                    }
                } else {
                    if (y >= y2 && y < y1) {
                        int x = (y - y1) * (polygon[j].x - polygon[i].x) / (y2 - y1) + polygon[i].x;
                        intersections.push_back(x);
                    }
                }
            }

            sort(intersections.begin(), intersections.end());
            int m = intersections.size();

            glColor3f(0, 1, 0);
            for(int i = 0; i < m; i += 2) {
                int x1 = intersections[i];
                int x2 = intersections[i + 1];
                glBegin(GL_LINES);
                    glVertex2i(x1, y);
                    glVertex2i(x2, y);
                glEnd();

                glFlush();
            }
        }

        cout<<"Finished filling!\n";

        glFlush();
        polygon.clear();
    }
    }



static void key(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'b':
            glutMouseFunc(boundaryFill);
            break;

        case 'f':
            glutMouseFunc(floodFill);
            break;

        case 'y':
            glutMouseFunc(yx);

    }

    glutPostRedisplay();
}


int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitWindowSize(MAXWIDTH,MAXHEIGHT);
    glutInitWindowPosition(10,10);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutCreateWindow("Polygon Filling");

    cout<<"b:Boundary Filling\nf:Flood Filling\ny:YX Filling";


    init();
    glutDisplayFunc(display);
    glutKeyboardFunc(key);

    glutMainLoop();

    return EXIT_SUCCESS;
}
