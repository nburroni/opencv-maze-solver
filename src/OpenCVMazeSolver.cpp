/* Este c—digo es una modificaci—n del ejemplo del tutorial http://docs.opencv.org/master/dc/d16/tutorial_akaze_tracking.html
 *
 * El prop—sito de este programa es identificar una foto en un video, y obtener su posici—n y direcci—n, por ejemplo por medio de una homograf’a.
 *
 * Uso:
 * teseo orb|akaze ruta_del_video [ruta_de_la_imagen_modelo]
 *
 *
 * Argumentos
 * ../Archivos/Maquiavelo.mp4
 * ../Archivos/Maquiavelo.jpg
 *
 * LISTO
 *  - Modelo de argumentos:
 *  Si hay un video, se abre.  Si no, se usa directamente la webcam y se impide cambiar a modo video.
 *  Si hay una imagen se usa como referencia.  Si no, se usa una imagen vac’a o negra.
 *
 *  - Teclas:
 *  esc, q, s: salir
 *  w: webcam on/off
 *  r: captura imagen de referencia
 *  d: alterna detector

 * TODO
 * -
 *
 *
 */

#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/videoio.hpp"
#include <vector>
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;




// START SECCION MAZE SOLVER

std::vector<cv::Point> reducePoints(std::vector<cv::Point> points) {
	int size = 0;
	for (uint i = 0; i < points.size(); i++) {
		if(points[i].x == 0 && points[i].y == 0) break;
		else size++;
	}
	std::vector<cv::Point> reduced = std::vector<cv::Point>(size);
	for (int i = 0; i < size; i++) {
		reduced[i].x = points[i].x;
		reduced[i].y = points[i].y;
	}
	return reduced;
}

int firstZero(std::vector<cv::Point> points) {
	for (uint i = 0; i < points.size(); i++) {
		if(points[i].x == 0 && points[i].y == 0) return i;
	}
	return -1;
}

std::vector<cv::Point> removeZeros(std::vector<cv::Point> points) {
	for (uint i = 0; i < points.size(); i++) {
		if(points[i].x != 0 && points[i].y != 0) {
			int zi = firstZero(points);
			if (zi != -1) {
				points[zi] = points[i];
				points[i] = Point(0, 0);
			}
		}
	}
	return reducePoints(points);
}

std::vector<cv::Point> optimizePoints(std::vector<cv::Point> points, int margin) {
	std::vector<cv::Point> optimized = std::vector<cv::Point>(points.size());
	optimized[0] = points[0];
	int j = 0;
	for (uint i = 1; i < points.size() - 1; i++) {
		cv::Point curr = points[i];
		cv::Point next = points[i + 1];
		if (!(next.x < (curr.x + margin) && next.x > (curr.x - margin) && next.y < (curr.y + margin) && next.y > (curr.y - margin))) {
			std::cout << "Including: ";
			std::cout << next << std::endl;
			optimized[++j] = next;
		}
	}
	return optimized;
}

void printPoints(std::vector<cv::Point> points) {
	for (uint i = 0; i < points.size(); i++) {
		std::cout << points[i] << std::endl;
	}
}

bool isContained(cv::Point point, std::vector<cv::Point> points, int margin) {
	for (uint i = 0; i < points.size(); i++) {
		if (point.x > (points[i].x - margin) && point.x < (points[i].x + margin)
				&& point.y > (points[i].y - margin) && point.y < (points[i].y + margin)) {
			std::cout << "Checking ";
			std::cout << point << std::endl;
			return true;
		}
	}
	return false;
}

bool allChecked(std::vector<bool> checked) {
	for (uint i = 0; i < checked.size(); i++) {
		if (!checked[i]) return false;
	}
	return true;
}

std::vector<cv::Point> solveMaze(cv::Mat src) {
	// Cablear imagen
//	src = cv::imread("src/the-maze.jpg");

	int binarySensitivity = 230;

	vector<cv::Point> errorReturn = vector<cv::Point>(0);
    if (src.empty()) {
        std::cout << "No such image found!" << std::endl;
        return errorReturn;
    }
//    cv::imshow("Maze", src);
//    char choice = cv::waitKey(0);

    // to black and white
    cv::Mat bw;
    cv::cvtColor(src, bw, CV_BGR2GRAY);
//    cv::imshow("Maze", bw);
//    choice = cv::waitKey(0);

    // To binary image
    cv::threshold(bw, bw, binarySensitivity, 255, CV_THRESH_BINARY);
    cv::imshow("BIN: (e)rode (d)ilate", bw);
    char choice = cv::waitKey(0);
    if (choice == 'q') {
		// Termina
		return errorReturn;
    }

    cv::Mat kernel = cv::Mat::ones(2, 2, CV_8UC1);

    while(choice == 'e' || choice == 'd') {
		switch(choice) {
			case 'e':
				cv::erode(bw, bw, kernel);
				break;
			case 'd':
				cv::dilate(bw, bw, kernel);
				break;
		}

		cv::imshow("BIN: (e)rode (d)ilate", bw);
		choice = cv::waitKey(0);
	    if (choice == 'q') {
			// Termina
			return errorReturn;
	    }
    }

//    Maze Solving
    std::vector<std::vector<cv::Point> > contours;
	cv::findContours(bw, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	if (contours.size() != 2)
	{
		// "Perfect maze" should have 2 walls
		std::cout << "This maze has ";
		std::cout << contours.size();
		std::cout << " walls..." << std::endl;
		return errorReturn;
	}

	// Crea la variable path con todos 0s, y le dibuja uno de los dos contornos
	cv::Mat path = cv::Mat::zeros(src.size(), CV_8UC1);
	cv::drawContours(path, contours, 0, CV_RGB(255,255,255), CV_FILLED);
//	cv::imshow("Maze", path);
//	choice = cv::waitKey(0);
//    if (choice == 'q') {
//		// Termina
//		return errorReturn;
//    }

	// Crea el kernel con todos 1s
	cv::Mat dKernel = cv::Mat::ones(55, 55, CV_8UC1);
	cv::imshow("Press 'd' to dilate", path);
	char dilate = cv::waitKey(0);
    if (dilate == 'q') {
		// Termina
		return errorReturn;
    }
	while(dilate == 'd') {
		cv::dilate(path, path, dKernel); // Dilata el path
		cv::imshow("Press 'd' to dilate", path);
		dilate = cv::waitKey(0);
	    if (choice == 'q') {
			// Termina
			return errorReturn;
	    }
	}

	// Erosiona el path a una nueva imagen
	cv::Mat eKernel = cv::Mat::ones(20, 20, CV_8UC1);
	cv::Mat path_erode;
	cv::erode(path, path_erode, eKernel);
//	cv::imshow("Maze", path_erode);
	cv::absdiff(path, path_erode, path); // Resta el dilatado con el erosionado y guarda en path
//	choice = cv::waitKey(0);
    if (choice == 'q') {
		// Termina
		return errorReturn;
    }
//	cv::imshow("Maze", path);


	// Loop para erosionar el camino resultante
	cv::Mat kernel2 = cv::Mat::ones(9, 9, CV_8UC1);
	cv::Mat exitPath = path.clone();
	cv::imshow("Press 'e' to erode", exitPath);
	char c = cv::waitKey(0);
    if (c == 'q') {
		// Termina
		return errorReturn;
    }
	while(c == 'e') {
		cv::erode(exitPath, exitPath, kernel2);
		cv::imshow("Press 'e' to erode", exitPath);
		c = cv::waitKey(0);
	    if (c == 'q') {
			// Termina
			return errorReturn;
	    }
	}

	std::vector<cv::Mat> channels;
	cv::split(src, channels);
	channels[0] &= ~exitPath;
	channels[1] &= ~exitPath;
	channels[2] |= exitPath;
//	cv::imshow("Maze", exitPath);
//	choice = cv::waitKey(0);
//    if (choice == 'q') {
//		// Termina
//		return errorReturn;
//    }

	cv::Mat dst;
	cv::merge(channels, dst);
	cv::imshow("Maze", dst);
	choice = cv::waitKey(0);
    if (choice == 'q') {
		// Termina
		return errorReturn;
    }


//    Segunda parte

	std::vector<std::vector<cv::Point> > pathContours;
	cv::findContours(exitPath, pathContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	std::cout << "Path Contours: ";
	std::cout << pathContours.size() << std::endl;

	cv::Mat a = cv::Mat::zeros(src.size(), CV_8UC1);
	cv::drawContours(a, pathContours, 0, CV_RGB(255,255,255), CV_FILLED);
//	cv::imshow("Exit Path", a);
//	std::cout << "SIZE" << std::endl;
//	std::cout << pathContours.size() << std::endl;
//	cv::waitKey(0);

	std::vector<cv::Point> solution = pathContours[0];

	std::vector<cv::Point> points;
	points = std::vector<cv::Point>(500);

	points[0] = solution[0];
	int j = 0, margin = 5;
	bool maintainX = solution[0].x == solution[1].x;
	for (uint i = 0; i < solution.size() - 1; i++) {
		if (solution[i + 1].x == 0 && solution[i + 1].y == 0) {
			points[++j] = solution[++i];
			break;
		}
		if (maintainX) {
			if (solution[i + 1].x < points[j].x - margin || solution[i + 1].x > points[j].x + margin) {
				points[++j] = solution[i];
				points[++j] = solution[++i];
				maintainX = false;
			}
		} else {
			if (solution[i + 1].y < points[j].y  - margin || solution[i + 1].y > points[j].y + margin) {
				points[++j] = solution[i];
				points[++j] = solution[++i];
				maintainX = true;
			}
		}
	}
	for (uint i = 0; i < points.size() - 1; i++) {
		if (points[i + 1].x == 0 && points[i + 1].y == 0) {
			points[i + 1].x = points[i].x;
			points[i + 1].y = solution[i].y + 500;
		}
	}

	std::cout << "Path Points: ";
	std::cout << solution.size() << std::endl;

	points = reducePoints(points);
	points = reducePoints(optimizePoints(points, 3 * margin));
	printPoints(points);

	std::cout << "Solution Points: ";
	std::cout << points.size() << std::endl;

	// Dibujar solution
	cv::Mat sol = cv::Mat::zeros(src.size(), CV_8UC1);
	int myradius=3;
	for (uint i=0;i<points.size();i++) {
//    	    	std::cout << "Drawing ";
//    	    	std::cout << solution[i].x;
//    	    	std::cout << ";";
//    	    	std::cout << solution[i].y << std::endl;
		circle(sol,cvPoint(points[i].x,points[i].y),myradius,CV_RGB(255,255,255),.5,8,0);
	}
	cv::imshow("Solution", sol);
	choice = cv::waitKey(0);
    if (choice == 'q') {
		// Termina
		return errorReturn;
    }

    // Eliminar puntos
	for (uint j=0;j<points.size();j++) {
		sol = cv::Mat::zeros(src.size(), CV_8UC1);
		int myradius=5;
		for (uint i=0;i<points.size();i++) {
			CvScalar color = CV_RGB(255,255,255);
			if(j == i) color = CV_RGB(100,100,100);
			circle(sol,cvPoint(points[i].x,points[i].y),myradius,color,.5,8,0);
		}
		cv::imshow("Delete?", sol);
		std::cout << j;
		std::cout << " -> ";
		char ch = cv::waitKey(0);
		if (ch == 'q') {
			// Termina
			return errorReturn;
		} else if (ch == 'd') {
			points[j] = Point(0,0);
			std::cout << " deleted" << std::endl;
	    } else {
	    	std::cout << " kept" << std::endl;
	    }
	}
	std::cout << "Before reduce" << std::endl;
	printPoints(points);
	points = removeZeros(points);
	std::cout << "After reduce" << std::endl;
	printPoints(points);

	// Ordenar los puntos
	for (uint j=0;j<points.size();j++) {
		sol = cv::Mat::zeros(src.size(), CV_8UC1);
		int myradius=5;
		for (uint i=0;i<points.size();i++) {
			CvScalar color = CV_RGB(255,255,255);
			if(j == i) color = CV_RGB(100,100,100);
			circle(sol,cvPoint(points[i].x,points[i].y),myradius,color,.5,8,0);
		}
		cv::imshow("Choose position", sol);
		std::cout << j;
		std::cout << " -> ";
		char ch = cv::waitKey(0);
	    if (ch == 'q') {
			// Termina
			return errorReturn;
	    }
		int pos = ch - '0';
		std::cout << pos << std::endl;

		cv::Point aux = points[pos];
		points[pos] = points[j];
		points[j] = aux;
	}
	// Ordenar los puntos inteligentemente
	std::vector<bool> checked = std::vector<bool>(points.size());
	for (uint j=0;j<checked.size();j++) checked[j] = false;
	while (!allChecked(checked)) {
		for (uint j=0;j<points.size();j++) {
			if (checked[j]) continue;
			sol = cv::Mat::zeros(src.size(), CV_8UC1);
			int myradius=5;
			for (uint i=0;i<points.size();i++) {
				CvScalar color = CV_RGB(255,255,255);
				if(j == i) color = CV_RGB(100,100,100);
				circle(sol,cvPoint(points[i].x,points[i].y),myradius,color,.5,8,0);
			}
			cv::imshow("Choose position", sol);
			std::cout << j;
			std::cout << " -> ";
			char ch = cv::waitKey(0);
		    if (ch == 'q') {
				// Termina
				return errorReturn;
		    }
			int pos = ch - '0';
			std::cout << pos << std::endl;

			cv::Point aux = points[pos];
			points[pos] = points[j];
			points[j] = aux;

			checked[pos] = true;
		}
	}

	std::cout << "**************************" << std::endl;

	std::cout << "Before adjustment" << std::endl;
	printPoints(points);

	// Acomodar puntos
	for (uint j=0; j < points.size() - 1;j++) {
		cv::Point curr = points[j];
		cv::Point next = points[j + 1];
		if (next.x - curr.x < next.y - curr.y) {
			std::cout << next;
			std::cout << " -> ";
			points[j + 1].x = curr.x;
			std::cout << points[j + 1] << std::endl;
		}
		else {
			std::cout << next;
			std::cout << " -> ";
			points[j + 1].y = curr.y;
			std::cout << points[j + 1] << std::endl;
		}
	}

	std::cout << "After adjustment" << std::endl;
	printPoints(points);

    return points;
}

// END SECCION MAZE SOLVER


void c(string mensaje){cout << mensaje << endl;}	// Utilidad para mensajes de debug en consola
const float PI = 3.14159265358979323846264338328;	// Constante cosmol—gica, con decimales de sobra por si se quiere aumentar la precisi—n


/* Algunas variables son globales por conveniencia, para ser utilizadas desde funciones.
 * Algunas de estas variables se definen a continuaci—n, y otras inmediatamente antes de la funci—n que las requiere.
 */


/* Detectores: la aplicaci—n permite al usuario alternar entre dos "features detectors": AKAZE y ORB
 * La variable aKaze selecciona el detector a utilizar: true para akaze y false para orb
 * La funci—n cambiarDetector acomoda los valores de referencia compatibles con el detector.
 * Se debe invocar al inicializar y cada vez que el usuario cambia el detector con la tecla "d".
 */
bool aKaze = true;
Mat imagenReferencia, descriptoresReferencia, descriptoresReferenciaOrb, descriptoresReferenciaAkaze;
vector<KeyPoint> kpsReferencia, kpsReferenciaOrb, kpsReferenciaAkaze;
Ptr<AKAZE> akaze;
Ptr<ORB> orb;
char const * metodo;	// Nombre del mŽtodo
void cambiarDetector(bool aKaze){
	if(aKaze){
		kpsReferencia = kpsReferenciaAkaze;
		descriptoresReferencia = descriptoresReferenciaAkaze;
		metodo = "akaze";
	} else {
		kpsReferencia = kpsReferenciaOrb;
		descriptoresReferencia = descriptoresReferenciaOrb;
		metodo = "orb";
	}
}


/* La funci—n procesarImagenReferencia registra la imagen argumento en la variable imagenReferencia,
 * y obtiene sus keypoints y descriptores tanto para akaze como para orb
 * El argumento opcional anchoImagen determina el ancho de la imagen de referencia: la imagen argumento ser‡ redimensionada manteniendo la relaci—n de aspecto.
 */
vector<Point2f> flechaReferencia(2);
void procesarImagenReferencia(Mat img, Size tamanio=Size(0,0)){
	// Copia la imagen a la variable imagenReferencia
	if(tamanio.width)
		resize(img, imagenReferencia, tamanio);
	else
		imagenReferencia = img.clone();

	/* Define el segmento "flecha" desde el centro de la imagen hacia derecha.  Luego este segmento se proyecta con homograf’a.
	 * La "flecha" es un segmento orientado, consistente en un vector de dos puntos: el elemento 0 y el elemento 1.
	 * El elemento 0 se ubica en el centro de la imagen de referencia,
	 * el elemento 1 es la "punta de la flecha" (una punta abstracta, ya que nunca se dibuja como tal)
	 */
	int alto=imagenReferencia.rows;
	int ancho=imagenReferencia.cols;
	flechaReferencia[0] = Point2f(ancho/2, alto/2);
	flechaReferencia[1] = Point2f(ancho,   alto/2);

	// Ejecuta ambos detectores para tener los keypoints y descriptores de la imagen de referencia.  Cada detector tiene su propio conjunto de keypoints y descriptores.
	akaze->detectAndCompute(imagenReferencia, noArray(), kpsReferenciaAkaze, descriptoresReferenciaAkaze);
	orb  ->detectAndCompute(imagenReferencia, noArray(), kpsReferenciaOrb,   descriptoresReferenciaOrb  );
	cambiarDetector(aKaze);	// Inicializa las variables para el detector elegido

	imshow("referencia", imagenReferencia);
	cout<<"Keypoints en la imagen de referencia: "<<kpsReferencia.size()<<endl;
}

/* Alterna flujo de video de entrada (video y webcam)
 * Ambos, si est‡n disponibles deben estar inicializados.
 * videoPresente y webcamPresente indican si est‡n disponibles.
 *
 * flujoVideoEntrada es el video que se procesa, un puntero que apunta a video o webcam
 * video es el proveniente del archivo de video pasado como argumento
 * webcam es el video proveniente de la webcam
 */
VideoCapture *flujoVideoEntrada = NULL, webcam, video;
bool videoPresente = false;
bool webcamPresente = true;
void alternarEntradaVideo(){
	if(!flujoVideoEntrada){
		// Inicializaci—n
		if(videoPresente)
			flujoVideoEntrada = &video;
		else
			flujoVideoEntrada = &webcam;

	} else if(flujoVideoEntrada == &video && webcamPresente)
		flujoVideoEntrada = &webcam;
	else if(videoPresente)
		flujoVideoEntrada = &video;
}


/* kpsAPts convierte los keypoints argumento a puntos.
 * Recibe un vector de keypoints, y genera y devuelve un vector de puntos Point2f
 */
vector<Point2f> kpsAPts(vector<KeyPoint> keypoints){
    vector<Point2f> puntos;
    for(unsigned i=0; i<keypoints.size(); i++)
    	puntos.push_back(keypoints[i].pt);
    return puntos;
}

/* Variables que registran la orientaci—n
 * x e y son coordenadas del centro de la imagen de referencia, expresadas en p’xeles, sobre la imagen de entrada
 * angulo es la orientaci—n en grados.  0 grados corresponde a la orientaci—n hacia la derecha.
 * longitud de la flecha, en p’xeles.  Corresponde a la mitad del ancho de la imagen de referencia, encontrada sobre la imagen de entrada.
 * flecha es el segmento flechaReferencia, proyectada sobre la imagen de entrada.
 */
float angulo, x, y, longitud;
vector<Point2f> teseo;

int main(int argc, char **argv){
	// Inicializa variables

    FILE *file;
    file = fopen("/dev/ttyUSB0","w");
	char instruction = 's';
	char prevInstruction = 'a';
	bool run = false;
	float segmentDelta = 30;

	int segmentIndex = 0;
	int nextSegmentIndex = 1;
//	Point2f segments[5][2] = {
//			{Point2f(75, 100), Point2f(490, 100)},
//			{Point2f(490, 100), Point2f(490, 225)},
//			{Point2f(490, 225), Point2f(135, 225)},
//			{Point2f(135, 225), Point2f(135, 350)},
//			{Point2f(135, 350), Point2f(550, 350)}
//	};
//	int segmentsLength = 5;


//	 SACAR RESULTADO DEL MAZE SOLVER
	std::vector<cv::Point> segments = std::vector<cv::Point>(1);
	int segmentsLength = 1;



	char tecla;	// Recibe la tecla pulsada por el usuario
	//unsigned i;	// Bucles for
	bool hayHomografia;
    Mat imagenEntrada, imagenSalida, maze;	// Imagen del flujo de entrada, e imagen de salida para mostrar
    namedWindow("referencia");
    namedWindow("salida");//, WINDOW_NORMAL);	// Crea la ventana "salida" donde se mostrar‡ el video con marcas superpuestas

    /* En este punto se inicializan los detectores AKAZE y ORB
     * Al iniciar, la aplicaci—n usa AKAZE, pero durante la ejecuci—n se puede alternar a ORB con la tecla "d"
     */
	// Inicializar AKAZE
	akaze = AKAZE::create();
	akaze->setThreshold(3e-4);

	// Inicializar Orb
	orb = ORB::create();

	/* Procesando argumentos
	 * El argumento opcional que comienza con "v" indica la ruta del video a abrir.  Si se omite se utiliza el flujo de la webcam.
	 * Durante la ejecuci—n se puede alternar entre video y webcam con la tecla "w".
	 * El c—digo abre la webcam[0], no verifica si existe.  Al intentar cambiar el flujo de entrada a una webcam inexistente el programa aborta.
	 *
	 * El argumento opcional que comienza con "i" indica la ruta de la imagen de referencia, la imagen a localizar sobre el video.
	 * Si se omite se utiliza el primer cuadro del video.
	 * Durante la ejecuci—n, con la tecla "r" se captura el cuadro actual para utilizar como imagen de referencia.
	 */
	bool hayImagenReferencia = false;
	for(int i=1; i<argc; i++)
		if(argv[i][0] == 'v'){
			// Archivo de video
		    video.open(argv[i]+1);
		    if(!video.isOpened()) cout << "No se pudo abrir el archivo de video " << argv[i]+1 << endl;
		    else videoPresente = true;
		} else if(argv[i][0] == 'i'){
			imagenReferencia = imread(argv[i]+1);
			procesarImagenReferencia(imagenReferencia , Size(700, 700*imagenReferencia.rows/imagenReferencia.cols));
			hayImagenReferencia = true;
		}

	// ABRIR WEBCAM: 0 ES LA DE LA COMPU, SI PONGO 1 ES USB
	if(webcamPresente) webcam.open(0);
	alternarEntradaVideo();	// Inicializa flujoVideoEntrada para que apunte a video si est‡ disponible, de otro modo a webcam.

	if(!hayImagenReferencia){
		flujoVideoEntrada->read(imagenEntrada);
		procesarImagenReferencia(imagenEntrada);
	}

    // Dimensiones del flujo de video de entrada
    int videoAlto = flujoVideoEntrada->get(CV_CAP_PROP_FRAME_HEIGHT );
    int videoAncho = flujoVideoEntrada->get(CV_CAP_PROP_FRAME_WIDTH );
    cout<<"Alto: "<<videoAlto<<", Ancho: "<<videoAncho<<endl;



    // Las teclas + y - duplican o reducen a la mitad el tama–o de la imagen del video.  Cambian el valor de escalaVideo.
    float escalaVideo = 1;	// Factor para aumentar o reducir el tama–o del video
    Size tamanioVideo = Size(videoAncho, videoAlto);	// tamanioVideo tiene inicialmente el tama–o del video

    // Variables para la detecci—n de features para obtener la homograf’a
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // Variables para registrar la orientaci—n obtenida de la homograf’a
	Mat homografia;
	vector<Point2f> Points(vector<KeyPoint>);

    // Bucle principal
    while(true){

    	/* Lee un cuadro del flujo de video de entrada y lo registra en imagenEntrada
    	 * Cambia el tama–o del cuadro, proporcional a escalaVideo
    	 */
    	flujoVideoEntrada->read(imagenEntrada);
        if(escalaVideo != 1)
        	resize(imagenEntrada, imagenEntrada, tamanioVideo);


        /* Procesa el feature detector y descriptor extractor (detectAndCompute) del algoritmo elegido (akaze u orb)
         *
         */
        vector< vector<DMatch> > matches;
        Mat descriptores;
        vector<KeyPoint> kps;//, matched1, matched2;
        vector<Point2f> matchesReferencia, matchesEntrada;

        if(aKaze)
        	akaze->detectAndCompute(imagenEntrada, noArray(), kps, descriptores);
        else{
        	orb->setMaxFeatures(kpsReferencia.size());
        	orb->detectAndCompute(imagenEntrada, noArray(), kps, descriptores);
        }

        /* Aparea keypoints del cuadro actual con los encontrados en la imagen de referencia
         * Para cada keypoint de la imagen de referencia busca los dos mejores candidatos del cuadro actual.
         * El primero de ambos candidatos es siempre mejor que el segundo.  Si el primer candidato es mucho mejor que el segundo, lo selecciona.
         * Si no, descarta ambos porque no tiene manera de saber cu‡l de los dos es el que corresponde.
         */
    	if(descriptores.rows) matcher->knnMatch(descriptoresReferencia, descriptores, matches, 2);
        for(unsigned i=0; i < matches.size(); i++) if(matches[i][0].distance < 0.8f * matches[i][1].distance){
        		matchesReferencia.push_back(kpsReferencia[matches[i][0].queryIdx].pt);
        		matchesEntrada   .push_back(kps			 [matches[i][0].trainIdx].pt);
            }

        /* C‡lculo de la homograf’a.
         * Se requieren al menos 4 matches.
         * Aun as’, puede suceder que no se pueda encontrar una homograf’a para los matches, por eso se pregunta si !homografia.empty()
         * Habiendo homograf’a, se transforma la flechaReferencia para obtener la flecha convenientemente proyectada.
         * Finalmente se dibuja la flecha en imagenEntrada y se registran x, y y angulo.
         */
        if(matchesReferencia.size()>=4){	// Calcula homograf’a si hay puntos suficientes
            homografia = findHomography(matchesReferencia, matchesEntrada, RANSAC, 2.5);
//            cout << "Enough Matches" << endl;
			if(!homografia.empty()){
//				cout << "Unempty" << endl;
				// Hay homograf’a
				hayHomografia = true;

		        perspectiveTransform(flechaReferencia, teseo, homografia);
//		        cout << "Checkpoint #1" << endl;

				// Obtiene la posici—n y el ‡ngulo a partir de los puntos de la flecha
				x = teseo[0].x;
				y = teseo[0].y;
				angulo = -atan2(teseo[1].y-y, teseo[1].x-x)*180/PI;
				longitud = sqrt(x*x+y*y);
//
//				float segmentX = segments[segmentIndex][1].x;
//				float segmentY = segments[segmentIndex][1].y;

				float segmentX = segments[nextSegmentIndex].x;
				float segmentY = segments[nextSegmentIndex].y;


				if (x > segmentX - segmentDelta && x < segmentX + segmentDelta && y > segmentY - segmentDelta && y < segmentY + segmentDelta) {
					segmentIndex++;
					nextSegmentIndex = segmentIndex + 1;
				}

//				float anguloSegmento = -atan2(segments[segmentIndex][1].y - segments[segmentIndex][0].y, segments[segmentIndex][1].x - segments[segmentIndex][0].x)*180/PI;

				float anguloSegmento = -atan2(segments[nextSegmentIndex].y - segments[segmentIndex].y, segments[nextSegmentIndex].x - segments[segmentIndex].x)*180/PI;

				float anguloPrima;
//				cout << anguloSegmento << endl;

				//Calcular angulo de teseo respecto al segmento, asumiendo que el segmento esta a 0, 90, 180 o 270 grados de la horizontal
				if (anguloSegmento > -10 && anguloSegmento < 10) { // 0 grados
//					cout << "case 1" << endl;
					anguloPrima = atan2(teseo[1].y - y, teseo[1].x - x)*180/PI;
				} else if (anguloSegmento > 80 && anguloSegmento < 100) { // 90 grados
//					cout << "case 2" << endl;
					anguloPrima = -atan2(-teseo[1].x + x, -teseo[1].y + y)*180/PI;
				} else if (anguloSegmento > 170 || anguloSegmento < -170) { // 180 grados
//					cout << "case 3" << endl;
					anguloPrima = atan2(-teseo[1].y + y, -teseo[1].x + x)*180/PI;
				} else { // -90 grados
//					cout << "case 4" << endl;
					anguloPrima = -atan2(teseo[1].x - x, teseo[1].y - y)*180/PI;
				}

				//instruccion => manda f para que ande para adelante, si no l o r para que gire a la izq o der
				int dir = anguloPrima / std::abs(anguloPrima);
				if(!run) instruction = 's';
				else instruction = anguloPrima > -15 && anguloPrima < 15 ? 102 : 111 - dir * 3;

				if(segmentIndex >= segmentsLength) instruction = 's';

				fprintf(file,"%c\n", instruction);
				cout << instruction << endl;
//				instruction = 's';
//				fprintf(file,"%c\n", instruction);
//				cout << instruction << endl;

				// Dibuja todos los segmentos en el frame
				for (int i = 0; i < segmentsLength - 1; i++) {
					Scalar color;
					if (i == segmentIndex) color = Scalar(255, 0, 0);
					else color = Scalar(0, 255, 0);

//					line(imagenEntrada, segments[i][1], segments[i][0], color);
//					circle(imagenEntrada, segments[i][0], 10, color);

					line(imagenEntrada, segments[i + 1], segments[i], color);
					circle(imagenEntrada, segments[i], 10, color);
				}

				// Dibuja la flecha sobre la imagen
//				cout << "Drawing" << endl;
				line(imagenEntrada, teseo[0], teseo[1], Scalar(255, 255, 0));
				circle(imagenEntrada, teseo[0], longitud/10, Scalar(255, 255, 0));



				// Muestra las coordenadas al pie de la ventana, en la barra de estado
				char mensaje[256];
				sprintf(mensaje, "%s: (%05.1f, %05.1f) <%05.1f> l:%05.1f", metodo, x, y, angulo, longitud);
//				displayStatusBar("salida", mensaje);
			} else{
				cout << "Empty" << endl;
//				displayStatusBar("salida", "Fall— el c‡lculo de la posici—n");
//				hayHomografia = false;
			}
		} else{
//			displayStatusBar("salida", "Pocos puntos para determinar la posici—n");
//			hayHomografia = false;
//			cout << "Exiting due to small amount of points" << endl;
		}


        // Mostrar imagen
        imshow("salida", imagenEntrada);

        // Teclado
        tecla = waitKey(5)%256;
        switch(tecla){
        case 27:
        case 's':
        	run = !run;
        	break;
        case 'q':
        	// Termina
			fprintf(file,"%c\n", 's');
            fclose(file);
        	return 0;

        case 'w':
        	// Alterna entre webcam y video
        	alternarEntradaVideo();
        	break;

        case 'd':
        	// Alterna detector
        	cambiarDetector(aKaze = !aKaze);
        	break;
        case 'z':
        	segmentIndex = 0;
        	nextSegmentIndex = 1;
        	break;
        case 'r':
        	// Nueva referencia: toma el cuadro siguiente, pues imagenEntrada tiene una flecha dibujada
    		flujoVideoEntrada->read(imagenEntrada);
        	procesarImagenReferencia(imagenEntrada);
        	break;
        case 'l':
        	// Nueva referencia: toma el cuadro siguiente, pues imagenEntrada tiene una flecha dibujada
    		flujoVideoEntrada->read(maze);
    		segments = solveMaze(maze);
    		segmentsLength = segments.size();
    		segmentIndex = 0;
    		nextSegmentIndex = 1;
        	break;
        case '-':
        case '+':
        	if(tecla == '+')
        		escalaVideo *= 2;
        	else
            	escalaVideo /= 2;
        	tamanioVideo = Size(escalaVideo * videoAncho, escalaVideo * videoAlto);
        	break;
        }
    }
    return 0;
}

