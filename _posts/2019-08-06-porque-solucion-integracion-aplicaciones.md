---
layout: post
comments: true
title:  "Porque una solución de Integración de Aplicaciones Empresariales"
date:   2019-08-06 10:50:34 -0500
categories: [Arquitectura de Software]
tags: [architectura de software]
---
![imagen intro](/assets/2019-08-06-porque-solucion-integracion-aplicaciones/alina-grubnyak-ZiQkhI7417A-unsplash.jpg)

## Contexto
Sin importar la industria, en la medida que las empresas crecen, pasando de ser pequeñas empresas para ser medianas o gran empresa, su dependencia de aplicaciones es mayor. Dependiendo de la estrategia de la empresa y su modelo de negocio, las empresas tienen mayor o menor dependencia de software. Esta dependencia también se acentúa por otras fuerzas, como son el crecimiento del negocio, la automatización de procesos, la implementación de nuevos canales, el desarrollo de nuevos productos, la integración con terceros, etc. Entonces en poco tiempo, una empresa termina dependiendo de una cantidad importante de aplicaciones, con diferentes tecnologías (nuevas y legadas), diferentes plataformas y lenguajes de programación. Aun cuando los líderes de tecnología, desde una perspectiva de arquitectura, han procurado mantener un ecosistema empresarial tecnológico armonioso para sus intereses, las fuerzas externas a sus fueros son muchas veces más fuertes.

En un mundo ideal, podríamos plantear el apilamiento de todas las necesidades del negocio en un solo sistema de información. Sin embargo, esto no ha venido siendo viable. Tan pronto como se implementa un sistema de información empresarial, sea que este ha sido desarrollado en-casa o ha sido adquirido, ya se tiene una cola importante de nuevas necesidades. A veces esta cola termina siendo tan grande y lenta, que las áreas funcionales o promotores de nuevas iniciativas de negocio buscan soluciones de software listas para usar. Esta problemática ha sido detectada por los proveedores de software y es por ello que a la fecha el mercado de soluciones de software viene creciendo en el sentido de resolver necesidades de negocio especificas frente a sistemas empresariales todo en uno. De otro lado y en ciertas circunstancias, sucede que el poder de los lideres funcionales es tal, que determinan las aplicaciones con las que ellos prefieren trabajar, a veces por experiencias en otras empresas. Finalmente, otra fuerza que empuja a incrementar el número de aplicaciones viene expresada por la ley de Conway. [Conway's law](https://en.wikipedia.org/wiki/Conway%27s_law).  <cite>"organizations which design systems ... are constrained to produce designs which are copies of the communication structures of these organizations"</cite>. Esto es, así exista un empoderamiento fuerte por parte de los directores o arquitectos de tecnología, el diseño de sistemas termina produciendo diversas aplicaciones que reflejan la estructura de la organización.

Si a este contexto le sumamos las fuerzas externas a la organización, como las que viene trayendo la [economia digital](https://es.wikipedia.org/wiki/Econom%C3%ADa_digital). El contexto resultante, desde la perspectiva de aplicaciones y software, se hace más complejo aún. La economía digital exige:
* **Integración**. Para tomar ventaja de los servicios provistos por terceros.
* **Desintermediación**. Para poner los servicios más cerca al cliente
* **Innovación** Para entregar y adaptar productos y servicios en forma continua
* **Inmediatez** Para integrar información y servicios, sin importar la fuente, que permita servir de inmediato al cliente
* **globalización** Para aprovechar las ventajas que trae la economía de escala. 

Cada uno de estos cinco puntos está vinculado estrechamente al software y la información. Aprovechar estas, exige diversificar aún más el ecosistema de aplicaciones de cualquier organización, aumentando su número. En este contexto, muchas organizaciones de tecnología dentro de las empresas, no están preparadas ni organizacionalmente ni tecnológicamente para enfrentar sistemáticamente la creciente necesidad del número de aplicaciones.


## La importancia de una visión integrada

En toda empresa, donde las aplicaciones empresariales cumplen un rol clave, es usual encontrar un equipo de desarrollo de software. Sin importar la metodología de desarrollo que estos equipos usen (convencional, agile o una mixtura de ambos), la capacidad de dicho equipo rápidamente se copa. Este copamiento involuntariamente crea una inercia de trabajo donde las prioridades de arquitectura e integración de las aplicaciones se posponen. Otro efecto generado por esta dinámica de trabajo es expresado por el concepto [Deuda Técnica](https://en.wikipedia.org/wiki/Technical_debt). Las soluciones “rápidas” y limitadas generan deuda, que, si no son pagadas, hacen que cualquier implementación posterior sea más costosa.

Por ello la importancia de tener una visión integrada de las aplicaciones y su entorno. Esta es importante, no para decir que se cumple con una buena práctica en la entrega de aplicaciones, sino porque está vinculado estrechamente a la generación sostenida de valor al negocio. Esto se observa en el siguiente diagrama. El entorno en el que se desarrollan los negocios viene cambiando, de la misma manera sus objetivos. Esto implica que los principios de arquitectura y las prácticas de implementación de aplicaciones deben ser reformuladas con la finalidad de aumentar su alineamiento.

![vision integrada](/assets/2019-08-06-porque-solucion-integracion-aplicaciones/vision_integrada.png)

Por tanto, contar con una visión integrada y alineada a las fuerzas que otorgan ventaja competitiva a la empresa, no es una opción, es una necesidad.

## Necesidades de Integración 

Aun cuando cada empresa tiene un contexto diferente que determina sus necesidades de integración de aplicaciones, el diagrama siguiente pretende mostrar un resumen de elementos y fuerzas que empujan la necesidad de integración de aplicaciones.

![necesidades de integracion](/assets/2019-08-06-porque-solucion-integracion-aplicaciones/necesidades_integracion.png)

Ninguno de los elementos mostrados en el diagrama puede existir en forma aislada. Cuando se habilita un canal digital al cliente, se pone a su disposición la capacidad de ser un actor en el proceso de negocio de la empresa. Por lo tanto, estos canales deben tocar/acceder a información que es parte de un sistema central. De igual manera, si se quiere explotar un servicio o capacidad de un tercero (servicio en la nube o no), es necesario integrar información de un tercero que permita generar una mejor experiencia al cliente en términos de inmediatez y simplicidad. De otro lado y sin importar la rapidez con la que una empresa adopte soluciones de software modernas, siempre tendrá que administrar la coexistencia de aplicaciones legadas y aplicaciones modernas en forma consistente e integrada.


## Integrador de Aplicaciones

Una forma común e incipiente de resolver las necesidades de integración de aplicaciones es el uso centralizado de una base de datos. Esto es, cada aplicación responde de forma particular a una necesidad especifica de negocio, pero todas ellas, o bien confluyen en una sola base de datos o en una sola tecnología de base de datos. Entonces, la necesidad de compartir información y servicios entre las aplicaciones se realiza a través de la base de datos. Si bien esta es una solución rápida, es limitada para responder las fuerzas y necesidades de integración antes mencionadas.

Las soluciones de integración de aplicaciones requieren hacer frente no solo un número importante de aplicaciones, sino a su diversidad tecnológica inherente. Esto es, necesitan ser capaces de integrar diferentes protocolos de comunicación, lenguajes de programación y tecnologías en general, sea que están sean propietarias o abiertas.

Otra característica de las soluciones de integración de aplicaciones es que estas abordan la comunicación entre aplicaciones de una forma diferente. Comúnmente, cuando una aplicación A se aloja en el mismo entorno de otra aplicación B, la forma convencional en que la aplicación A consume o hace una llamada a la aplicación B es por medio de llamadas síncronas. Esto es, iniciada la llamada de la aplicación A hacia la aplicación B, la aplicación A queda bloqueada (esperando) hasta que la aplicación B responda o se produzca una respuesta de tiempo excedido. Naturalmente, en el contexto actual, esta forma de abordar la comunicación entre aplicaciones tiene sus limitaciones. Primero, la comunicación entre dos aplicaciones A y B, se realiza a través de medios necesariamente no fiables como Internet. Segundo, frente a grandes volúmenes de transaccionalidad y muchos enlaces de integración, tener en espera los hilos de comunicación entre aplicaciones, es costoso en términos de recursos y frágil en términos de resistencia a un error. Debido a esto, es que la integración entre aplicaciones suele hacerse por medio de llamadas asíncronas.

Para entender la diferencia entre las llamadas síncronas y asíncronas usaremos la siguiente analogía. Una llamada síncrona es como una llamada telefónica. Si al iniciar la llamada de A hacia B, B está lejos del teléfono, por 3 o 4 timbrados, A espera hasta que B responda. Una vez que B responde, ambas tienen que estar disponibles para que el mensaje fluya. Lo natural es que A y B queden bloqueados durante todo el tiempo de la comunicación. Esto es, sus capacidades no son usadas para realizar otras actividades. De otro lado, una llamada asíncrona es como un mensaje de WhatsApp. Una vez que A envía un mensaje a B, A no espera a que B este cerca al teléfono, para que el mensaje sea enviado. B puede que lea y responda en dicho momento o tiempo después. Sin embargo, todo ese tiempo A está dedicando sus capacidades a realizar otras actividades. Una vez que A recibe el mensaje de respuesta de B, A retoma el contexto de la conversación, y continua su comunicación con B. 

La comunicación asíncrona, como se puede inferir, es más versátil en medios de comunicación menos fiables como Internet, pues que algo demore un segundo más o menos no altera el propósito de la comunicación. También es más versátil para adaptarse a la capacidad de los entes que emiten/reciben mensajes. Si uno tiende a ser aquella persona que envía/responde “rápido” a los mensajes de WhatsApp, es posible que sus contactos le escriban con frecuencia. Mientras que, si uno suele posponer su interacción con WhatsApp, con el tiempo, tendrá menos interacciones de mensajería por este canal.

Hemos mencionado que un integrador de aplicaciones necesita abordar hasta ahora dos situaciones. Hacer comunicar a dos aplicaciones totalmente diferentes y gestionar las llamadas entre ellas de una forma asíncrona. Solo estas dos situaciones, generan una multiplicidad de escenarios de integración. La recurrencia de estos escenarios en el ámbito empresarial, a través del tiempo, ha permitido modelar, a los especialistas en este campo, patrones de integración empresarial [Enterprise Integration Patterns](https://www.enterpriseintegrationpatterns.com/) que son de mucha utilidad a la hora de diseñar una solución de integración. Como dice el dicho, para que reinventar la pólvora. 

![patrones de integracion](/assets/2019-08-06-porque-solucion-integracion-aplicaciones/patterns.png)


Naturalmente estos patrones de integración son atómicos, cada uno de ellos cumple una función muy específica. Las soluciones de integración de aplicaciones disponibles en el mercado han implementado estos patrones usando alguna tecnología especifica. Cuando esto sucede, hablamos de tener una solución de integración de aplicaciones o un integrador de aplicaciones. Una vista para entender el rol de estas soluciones en el ámbito de aplicaciones empresariales es como se muestra en el diagrama siguiente.

![integrador de aplicaciones](/assets/2019-08-06-porque-solucion-integracion-aplicaciones/integrador.png)

Si bien el diagrama muestra dos marcas, como SAP y Saleforce, son solo referenciales. Una solución de integración no está limitada a una tecnología especifica. Una solución de este tipo permite integrar la comunicación entre dos o más aplicaciones, haciendo uso de diversas capacidades. Por ejemplo, la capacidad de transformación le permite transformar el formato de un mensaje específico para que este pueda ser interpretado por otra aplicación. La capacidad de enrutamiento le permite enrutar el mensaje a un punto B o C dependiendo de alguna condición o regla. La capacidad de orquestación, le permite componer varias llamadas en una sola llamada y articular tu ejecución.


## Soluciones de Integración

Las soluciones de integración de aplicaciones en el mercado son muchas. Existen soluciones de integración bastante grandes como las de Oracle, IBM, RedHat, etc. Sin embargo, tres soluciones que están teniendo bastante tracción en los últimos años son: Spring Integration, Apache Camel y MuleSoft. Aun cuando las tres soluciones implementan los patrones de integración empresarial antes mencionados, las soluciones no necesariamente tienen el mismo alcance y enfoque. 

* **Spring Integration** es la solución de integración de aplicaciones promovida por Spring. Cuando uno ya trabaja con Spring para algún otro proyecto, el uso de Spring Integration sería una gran alternativa, pues la curva de aprendizaje es más corta. Spring Integration es open source. 

* **Apache Camel** es soportado por la fundación de software Apache y también es open source. Los que usan esta solución, reconocen su amplia gama de conectores para diferentes aplicaciones y su simple lenguaje en la implementación. Cabe mencionar que Apache Camel viene siendo usado como parte de la suite comercial Red Hat Fuse. 

* **MuleSoft** no es open source pero tiene una versión comunidad que se puede usar. Su popular versión comercial, viene con herramientas que modelamiento para el diseño visual de las integraciones. 

En Internet uno puede encontrar comparaciones más detalladas de estas tres soluciones, como en [which-integration-framework](https://dzone.com/articles/which-integration-framework)

![soluciones de integracion](/assets/2019-08-06-porque-solucion-integracion-aplicaciones/providers.png)


## Desafíos para iniciar un proyecto de integración

Implementar una solución de integración de aplicaciones es una tarea compleja y ardua. Empezar enfocado en una integración de limitado alcance, para progresivamente ir aprovechando mejor la solución puede ser una estrategia por usar. Durante este camino, naturalmente existirán desafíos con los que hay que lidiar:

* **Repensar la organización del equipo de desarrollo de software**. Típicamente contar con un rol de arquitecto de software o aplicaciones ayuda bastante para tener una visión integrada de las aplicaciones y asegurar que la tecnología responda a las necesidades actuales y futuras del negocio.

* **Cambio en los paradigmas de programación**. Construir soluciones que exploten llamadas asíncronas (que hemos visto), requiere de cambiar el paradigma de programación. Los desarrolladores y arquitectos de software tienen que lidiar de forma diferente con la concurrencia, transaccionalidad y manejo de los errores en las aplicaciones.

* **Soluciones propietarias**. Aun cuando el ecosistema de software tiende al uso de estándares, existen aplicaciones legadas o aplicaciones propietarias, donde es todo un desafío encontrar conectores o adaptadores para hacerlos parte una solución de integración.

* **La solución se convierte en un componente crítico**. Dado que la solución de integración comunica las aplicaciones críticas en la organización, su rol también se vuelve crítico. En necesario ver con cautela su disponibilidad para enfrentar escenarios de contingencia que aseguren la continuidad del negocio. 

## Referencias
* [Enterprise Integration Patterns](https://www.enterpriseintegrationpatterns.com/)
* [Patterns and Best Practices for Enterprise Integration](https://www.amazon.com/o/asin/0321200683/)
* [Building microservices](https://www.oreilly.com/library/view/building-microservices/9781491950340/)
* [Microservice Architecture](https://microservices.io/)
* [Economía digital](https://es.wikipedia.org/wiki/Econom%C3%ADa_digital)
