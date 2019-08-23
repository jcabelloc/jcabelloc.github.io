---
layout: post
comments: true
title:  "Introducción a Microservicios"
date:   2019-08-22 11:50:34 -0500
categories: [Arquitectura de Software]
tags: [architectura de software, microservicios]
---
![imagen intro](/assets/2019-08-23-introduccion-microservicios/art-artificial-intelligence-blackboard-355948.jpg)

## Introducción
Este articulo pretende ayudarnos a comprender la Arquitectura de Microservicios aplicadas al desarrollo de aplicaciones empresariales. Veremos que fuerzas motivaron su surgimiento, así como que patrones e implementaciones existen hoy.

## Que son los microservicios

Microservicios es un "nuevo" paradigma a la hora de definir la arquitectura de una aplicación de software. Esto es, existen nuevas prácticas y patrones de arquitectura a la hora de descomponer una aplicación empresarial en "pequeños" servicios. Adoptando microservicios uno logra tener servicios de software que:

* Esten **distribuidos**, logrando que su operación en conjunto cubra las funciones de una aplicación completa.
* Presenten **bajo acomplamiento**, usando protocolos no propietarios como REST y encapsulando su implementación. Entonces la tecnología subyacente de cada servicio es irrelevante.
* Tengan **limitada responsabilidad**, ejecutando solo un pequeño numero de tareas definidas. 

Como se puede apreciar en el diagrama siguiente, una aplicación ficticia ha sido descompuesta en tres servicios. El servicio de cuentas, el servicio de inventarios y el servicio de entrega. Cada servicio se aloja típicamente en un servidor distinto y cada uno de ellos tiene su propio almacenamiento de datos. En una aplicación con arquitectura tradicional, también llamada monolítica, estos servicios junto a otros serían los módulos o paquetes, todos integrados en un solo sistema

![arquitectura de microservicios](/assets/2019-08-23-introduccion-microservicios/microservice_architecture.png)

Imagen tomada de: [microservicios.io](https://microservices.io/patterns/microservices.html)


## Porque cambiar?
En la era de Internet, donde los clientes valoran la inmediatez, la experiencia digital y una constante mejora en los servicios que reciben, los negocios esperan de sus organizaciones:
* Que se acabe con la complejidad que limita evolucionar su diferencia competitiva.
* Respuestas más rápidas al cliente
* Disponibilidad continua de sus servicios
* Capacidad de escalabilidad ante demandas inesperadas


Sin embargo, la forma en que se han venido construyendo las aplicaciones de software que soportan el negocio presentan síntomas que limitan las nuevas fuerzas al hacer negocios. Ver diagrama siguiente:
![sistema monolitico](/assets/2019-08-23-introduccion-microservicios/sistema_monolitico.png)


## Consideraciones claves para adoptar microservicios
Adoptando una arquitectura de microservicios genera los siguientes beneficios
* Los equipos de desarrollo de software llegan a ser más productivos
* La aplicación en su conjunto se forma por servicios más simples y fáciles de mantener.
* Se puede responder al cambio con mayor facilidad y flexibilidad
* Los cambios se introducen en forma continua

Sin embargo, para cosechar estos beneficios es importante considerar aspectos claves a la hora de adoptar microservicios:
* Servicios de **tamaño adecuado**. Esto es, cada servicio debe cumplir bien una sola responsabilidad. Un servicio con muchas responsabilidades hace compleja su mantenibilidad. 
* Servicios con **independencia de la infraestructura**. Esto es, los servicios deben poder alojarse en cualquier ubicación física y deben poder escalar (up/down) en forma transparente.
* Entornos **preparados ante fallos**. Esto es, ante un problema con un servicio, los clientes de los servicios deben tener rutas alternativas definidas. 
* Servicios **repetibles**. Esto es, cada vez que una nueva instancia de un servicio se activa, se basa en el mismo código y configuración.



## Los patrones de diseño
Como es sabido, la recurrencia para resolver un problema usando software, genera un patrón de diseño. La adopción constante de una arquitectura de microservicios nos ha dejado al día de hoy de patrones de diseño bastantes definidos a la hora de adoptar microservicios. Estos son:

1. **Patrones de construccion core**. Estos patrones nos ayudan a: 
    * Establecer el adecuado tamaño y responsabilidad de cada uno de los servicios. 
    * Definir los protocolos de comunicación de los servicios
    * Establecer los mecanismos para gestionar la configuración de los servicios.

2. **Patrones de Enrutamiento**. Estos patrones nos ayudan a: 
    * Abstraer la ubicación física de red del servicio a la hora de ser consumido. 
    * Proveer un solo punto de acceso hacia los servicios. 
3. **Patrones de Tolerancia a fallos**. Estos patrones nos ayudan a:
    * Balancear la carga cuando se realizan las llamadas desde los clientes. 
    * Evitar que clientes hagan llamadas a servicios que presentan fallas. 
    * Proveer respuestas alternativas antes servicios que presentan fallas.
4. **Patrones de Seguridad**. Estos patrones nos ayudan a:
    * Gestionar la autenticación y autorización  al llamar a servicios protegidos.
    * Gestionar y propagar los tokens de acceso obtenidos.
5. **Patrones de trazabilidad y logging**. Estos patrones nos ayudan a: 
    * Correlacionar los "logs" producidos por los diferentes servicios. 
    * Agrupar los logs para una mejor trazabilidad. 
    * Visualizar la trazabilidad de los eventos en forma integrada. 
6. **Patrones de despliegue**. Nos ayudan a: 
    * Definir procesos de construcción y despliegue continuo. 
    * Tratar las configuraciones de infraestructura tecnológica como código.
    * Generar contenedores de los servicios que son inmutables a cambios cada vez que se despliegan.



## Tecnologia existente

Los patrones son el marco de especificación que necesita estar soportado en alguna implementación. En la actualidad existen diversas implementaciones de los patrones indicados, muchos de ellos inclusive antes de la existencia de arquitectura de microservicios. En el entorno Java, **Spring Boot** y **Spring Cloud** vienen siendo usado con mucho éxito al momento de implementar una arquitectura de microservicios. 

* **Spring Boot** hace que el proceso que construcción y configuración de servicios REST sea una labor más simple,
* **Spring Cloud** integra una colección de tecnologías open source de compañías como Netflix y HashiCorp que simplifican la gestión de los servicios en el enfoque de microservicios.


En el siguiente diagrama presentamos la pila tecnológica en el entorno Java y Spring Framework.  
![tecnologias de microservicios](/assets/2019-08-23-introduccion-microservicios/tecnologia_microservicios.png)



## Cuando no adoptar microservicios

A pesar de los grandes beneficios que trae la adopción de una arquitectura de microservicios, su adopción no es recomendable en los siguientes escenarios.
* La **aplicación** de software es **departamental**. Esto es, descomponer sus funciones no justifica el esfuerzo.
* **No se invierte** en los procesos y herramientas que **automatizan y monitorean** la gestión de dependencias, la compilación, las pruebas, la entrega y el despliegue del software.
* La aplicación de software soporta procesos funcionales: **maduros**, de **baja necesidad de cambios** y de **baja necesidad de escalamiento**. Por ejemplo, un módulo contable. 
* La aplicación de software requiere de **agregar o transformar** data involucrando **muchas fuentes de datos**. Por ejemplo, un tablero de control empresarial. 

## Conclusiones
Los microservicios nos ofrecen una alternativa de arquitectura de software a la hora de implementar aplicaciones empresariales. Su adopción requiere de introducir nuevas prácticas, procesos y herramientas que ayuden a incrementar el éxito de su implementación.

## Referencias
* [Microservice Arquitecture](https://microservices.io/)
* [Spring Microservices in Action](https://www.manning.com/books/spring-microservices-in-action)
* [Building Microservices](https://www.oreilly.com/library/view/building-microservices/9781491950340/)