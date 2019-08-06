---
layout: post
comments: true
title:  "Porque una solucion de Integración de Aplicaciones Empresariales"
date:   2019-08-06 10:50:34 -0500
categories: [Arquitectura de Software]
tags: [architectura de software]
---

## Contexto
Sin importar la industria, en la medida que las empresas crecen, pasando de ser pequeñas empresas para ser medianas o gran empresa, su dependencia de aplicaciones es mayor. Dependiendo de la estrategia de la empresa y su modelo de negocio, las empresas tienen mayor o menor dependencia de software. Esta dependencia tambien se acentua por otras fuerzas, como son el crecimiento del negocio, la automatizacion de procesos, la implementacion de nuevos canales, el desarrollo de nuevos productos, la integracion con terceros, etc. Entonces en poco tiempo, una empresa termina dependiendo de una cantidad importante de aplicaciones, con diferentes tecnologias (nuevas y legadas), diferentes plataformas y lenguajes de programacion. Aun cuando los lideres de tecnologia, desde una perspectiva de arquitectura, han procurado mantener un ecosistema empresarial tecnologico armonioso para sus intereses, las fuerzas externas a sus fueros son muchas veces mas fuertes. 

En un mundo ideal, podriamos plantear el apilamiento de todas las necesidades del negocio en un solo sistema de informacion, sin embargo esto no ha venido siendo viable. Tan pronto como se implementa un sistema de informacion empresarial, sea que este ha sido desarrollado en-casa o ha sido adquirido, ya se tiene una cola importante de nuevas necesidades. A veces esta cola termina siendo tan grande y lenta, que las areas funcionales o promotores de nuevas iniciativas de negocio buscan soluciones de software listas para usar. Esta problematica ha sido detectada por los proveedores de software y es es por ello que a la fecha el mercado de soluciones de software viene creciendo en el sentido de resolver necesidades de negocio especificas frente a sistemas empresariales todo en uno. De otro lado y en ciertas circunstancias, sucede que el poder de los lideres funcionales es tal, que determinan las aplicaciones con las que ellos prefieren trabajar, a veces por experiencias en otras empresas. Finalmente, otra fuerza que empuja a incrementar el numero de aplicaciones viene expresada por la ley de Conway. [Conway's law](https://en.wikipedia.org/wiki/Conway%27s_law).  <cite>"organizations which design systems ... are constrained to produce designs which are copies of the communication structures of these organizations"</cite>. Esto es, asi exista un empoderamiento fuerte por parte de los directores o arquitectos de tecnologia, el diseño de sistemas termina produciendo diversas aplicaciones que reflejan la estructura de la organizacion.


Si a este contexto le sumamos las fuerzas externas a la organizacion, como las que viene trayendo la [economia digital](https://es.wikipedia.org/wiki/Econom%C3%ADa_digital). El contexto resultante, desde la perpectiva de aplicaciones y software, se hace mas complejo aun. La economia digital exige 1) integracion, para tomar ventaja de los servicios provistos por terceros, 2) desintermediacion, para poner los servicios mas cerca al cliente, 3) innovacion, para entregar y adaptar productos y servicios en forma continua, 4) inmediatez, para integrar informacion y servicios, sin importar la fuente, que permita servir de inmediato al cliente, 5) globalizacion, para aprovechar las ventajas que trae la economia de escala. Cada uno de estos cinco puntos esta vinculado estrechamente al software y la informacion. Aprovechar estas exige diversificar aun mas el ecosistema de aplicaciones de cualquier organizacion, aumentando su numero. En este contexto, muchas organizaciones de tecnologia dentro de las empresas, no estan preparadas ni organizacionalmente ni tecnologicamente para enfrentar sistematicamente la creciente necesidad del numero de aplicaciones. 


## La importancia de una visión integrada

En toda empresa, donde las aplicaciones empresariales cumplen un rol clave, es usual encontrar un equipo de desarrollo de software. Sin importar la metodologia de desarrollo que estos equipos usen (convencional, agile o una mixtura de ambos), la capacidad de dicho equipo rapidamente se copa. Este copamiento involutariamente crea una inercia de trabajo donde las prioridades de arquitectura e integracion de las aplicaciones se posponen. Otro efecto generado por esta dinamica de trabajo es expresado por el concepto [Deuda Tecnica](https://en.wikipedia.org/wiki/Technical_debt). Las soluciones "rapidas" y limitadas generan deuda, que si no son pagadas, hacen que cualquier implementacion posterior sea mas costosa. 

Por ello la importancia de tener una vision integrada de las aplicaciones y su entorno. Esta es importante, no para decir que se cumple con una buena practica en la entrega de aplicaciones, sino porque esta vinculado estrechamente a la generacion sostenida de valor al negocio. Esto se observa en el siguiente diagrama. El entorno en el que se desarrollan los negocios viene cambiando, de la misma manera sus objetivos. Esto implica que los principios de arquitectura y las practicas de implementacion de aplicaciones deben ser replanteadas con la finalidad de aumentar su alineamiento. 

![png](/assets/2019-08-06-porque-solucion-integracion-aplicaciones/vision_integrada.png)

Por tanto, contar con una vision integrada y alineada a las fuerzas que otorgan ventaja competitiva a la empresa, no es una opcion, es una necesidad. 

## Necesidades de Integración 

Aun cuando cada empresa tiene un contexto diferente que determina sus necesidades de integracion de aplicaciones, el diagrama siguiente pretende mostrar un resumen de elementos y fuerzas que empujan la necesidad de integración de aplicaciones. 

![png](/assets/2019-08-06-porque-solucion-integracion-aplicaciones/necesidades_integracion.png)

Ninguno de los elementos mostrados en el diagrama puede existir en forma aislada. Cuando se habilita un canal digital al cliente, se pone a su disposicion la capacidad de ser un actor en el proceso de negocio de la empresa. Por lo tanto, estos canales deben tocar/acceder a informacion que es parte de un sistema central. De igual manera, si se quiere explotar un servicio o capacidad de un tercero (servicio en la nube o no), es necesario integrar informacion de un tercero que permita generar una mejor experiencia al cliente en terminos de inmediatez y simplicidad. De otro lado, sin importar la rapidez con la que una empresa adopte soluciones de software modernas, siempre tendra que administrar la coexistencia de aplicaciones legadas y aplicaciones modernas en forma consistente e integrada.


## Integrador de Aplicaciones

Una forma comun e incipiente de resolver las necesidades de integracion de aplicaciones es el uso centralizado de una base de datos. Esto es, cada aplicacion responde de forma particular a una necesidad especifica de negocio, pero todas ellas o bien confluyen en una sola base de datos o en una sola tecnologia de base de datos. Entonces, la necesidad de compartir informacion y servicios entre las aplicaciones se realiza a traves de la base de datos. Si bien esta es una solucion rapida, es limitada para responder las fuerzas y necesidades de integracion antes mencionadas. 

Las soluciones de integracion de aplicaciones requieren hacer frente no solo un numero importante de aplicaciones, sino a su diversidad tecnologica inherente. Esto es, necesitan ser capaces de integrar diferentes protocolos de comunicacion, lenguajes de programacion y tecnologias en general, sea que estan sean propietarias o abiertas. 

Otra caracteristica de las soluciones de integracion de aplicaciones es que estas abordan la comunicacion entre aplicaciones de una forma diferente. Comunmente, cuando una aplicacion A se aloja en el mismo entorno de otra aplicacion B, la forma convencional en que la aplicacion A consume o hace una llamada a la aplicacion B es por medio de llamadas sincronas. Esto es, iniciada la llamada de la aplicacion A a la aplicacion B, la aplicacion A queda bloqueada (esperando) hasta que la aplicacion B responda o se produzca un respuesta de tiempo excedido. Natualmente, en el contexto actual, este forma de abordar la comunicacion entre aplicaciones, tiene sus limitaciones. Primero, la comunicacion entre dos aplicaciones A y B, se realiza a traves de medios necesariamente no fiables como Internet. Segundo, frente a grandes volumenes de transaccionalidad y muchos enlaces de integracion, tener en espera los hilos de comunicacion entre aplicaciones, es costoso en terminos de recursos y fragil en terminos de resistencia a un error. Debido a esto, es que la integracion entre aplicaciones suele hacerse por medio de llamadas asincronas.

Para entender la diferencia entre las llamadas sincronas y asincronas usaremos la siguiente analogia. Un llamada sincrona es como una llamada telefonica. Si al iniciar la llamada de A a B, B esta lejos del telefono, por 3 o 4 timbrados, A espera hasta que B responda. Una vez que B responde, ambas tienen que estar disponibles para que el mensaje fluya. Lo natural es que A y B queden bloqueados durante todo el tiempo de la comunicacion. Esto es, sus capacidades no son usadas para realizar otras actividades. De otro lado, una llamada asincrona es como un mensaje de Whatsapp. Una vez que A envia un mensaje a B, A no espera a que B este cerca al telefono, para que el mensaje sea enviado. B puede que lea y responda en dicho momento o tiempo despues. Sin embargo, todo ese tiempo A esta dedicando sus capacidades a realizar otras actividades. Una vez que A recibe el mensaje de B, A retoma el contexto de la conversacion, y continua su comunicacion con B. La comunicacion asincrona, como se puede inferir, es mas versatil en medios de comunicacion menos fiables como Internet, pues que algo demore un segundo mas o menos no altera el proposito de la comunicacion. Tambien es mas versatil para adaptarse a la capacidad de los entes que emiten/reciben mensajes. Si uno tiende a ser aquel persona que envia/responde "rapido" a los mensajes de Whatsapp, es posible que sus contactos le escriban con frecuencia. Mientras que si uno suele posponer su interaccion con Whatsapp, con el tiempo, tendra menos interacciones de mensajeria por este canal. 

Hemos mencionado que un integrador de aplicaciones, necesita abordar hasta ahora dos situaciones. Hacer comunicar a dos aplicaciones totalmente diferentes y gestionar las llamadas entre ellas de una forma asincrona. Solo estas dos situaciones, generan una multiplicidad de escenarios de integracion. La recurrencia de estos escenarios en el ambito empresarial, a traves del tiempo, ha permitido modelar, a los especialistas en este campo, patrones de integracion empresarial [Enterprise Integration Patterns](https://www.enterpriseintegrationpatterns.com/) que son de mucha utilidad a la hora de diseñar una solución de integración. Como dice el dicho, para que reinventar la polvora. 

![png](/assets/2019-08-06-porque-solucion-integracion-aplicaciones/patterns.png)


Naturalmente estos patrones de integracion son atomicos, cada uno de ellos cumple una funcion muy especifica. Las soluciones de integracion de aplicaciones disponibles en el mercado, han implementado estos patrones usando alguna tecnologia especifica. Cuando esto sucede, hablamos de tener una solucion de integracion de aplicaciones o un integrador de aplicaciones. Una vista para entender el rol de estas soluciones en el ambito de aplicaciones empresariales es como se muestra en el diagrama siguiente.

![png](/assets/2019-08-06-porque-solucion-integracion-aplicaciones/integrador.png)

Si bien el diagrama muestra dos marcas, como SAP y Saleforce, son solo referenciales. Una solucion de integracion no esta limitada a una tecnologia especifica. Una solucion de este tipo permite integrar la comunicacion entre dos o mas aplicaciones, haciendo uso de diversas capacidades. Por ejemplo, la capacidad de transformacion le permite transformar el formato de un mensaje especifico para que este pueda ser interpretado por otra aplicacion. La capacidad de enrutamiento le permite enrutar el mensaje a un punto B o C dependiendo de alguna condicion o regla. La capacidad de orquestacion, le permite componer varias llamadas en una sola llamada y articular tu ejecucion. 


## Soluciones de Integración

Las soluciones de integracion de aplicaciones en el mercado son muchas. Existen soluciones de integracion bastante grandes como las de Oracle, IBM, RedHat, etc. Sin embargo, tres soluciones que estan teniendo bastante traccion en los ultimos años son: Spring Integration, Apache Camel y MuleSoft. Aun cuando las tres soluciones implementan los patrones de integracion empresarial antes mencionados, las soluciones no necesariamente tienen el mismo alcance y enfoque. Spring Integration es la solucion de integracion de aplicaciones promovida por Spring. Cuando uno ya trabaja con Spring para algun otro proyecto, el uso de Spring Integration seria una gran alternativa, pues la curva de aprendizaje es mas corta. Spring Integration es open source. Apache Camel es soportado por la fundacion de software Apache y tambien es open source. Los que usan esta solucion, reconocen su amplia gama de conectores para diferentes aplicaciones y su simple lenguage en la implementacion. Cabe mencionar que Apache Camel viene siendo usado como parte de la suite comercial Red Hat Fuse. MuleSoft no es open source pero tiene una version comunidad que se puede usar. Su popular version comercial, viene con herramientas que modelamiento para el diseño visual de las integraciones. En Internet uno puede encontrar comparaciones mas detalladas de estas tres soluciones, como en [which-integration-framework](https://dzone.com/articles/which-integration-framework)

![png](/assets/2019-08-06-porque-solucion-integracion-aplicaciones/providers.png)


## Desafios para iniciar un proyecto de integración

Implementar una solucion de integracion de aplicaciones es una tarea compleja y ardua. Empezar enfocado en una integracion de limitado alcance, para progresivamente ir aprovechando mejor la solucion puede ser una estrategia a usar. Durante este camino, naturalmente existiran desafios con los que hay que lidear:

* **Repensar la organizacion del equipo de desarrollo de software**. Tipicamente contar con un rol de arquitecto de software o aplicaciones ayuda bastante para tener una vision integrada de las aplicaciones y asegurar que la tecnologia responda a las necesidades actuales y futuras del negocio. 

* **Cambio en los paradigmas de programacion**. Construir soluciones que exploten llamadas asincronas (que hemos visto), requiere de cambiar el paradigma de programacion. Los desarrolladores y arquitectos de software tienen que lidear de forma diferente con la concurrencia, transaccionalidad y manejo de los errores en las aplicaciones. 

* **soluciones propietarias**. Aun cuando el ecosistema de software tiende al uso de estandares, existen aplicaciones legadas o aplicaciones propietarias, donde es todo un desafio encontrar conectores o adaptadores para hacerlos parte una solucion de integracion.

* **La solucion se convierte en un componente critico**. Dado que la solucion de integracion comunica las aplicaciones criticas en la organizacion, su rol tambien se vuelve critico. En necesario ver con cautela su disponibilidad para enfrentar escenarios de contingencia que aseguren la continuidad del negocio. 

## Referencias
* Enterprise Integration Patterns: https://www.enterpriseintegrationpatterns.com/
* Patterns and Best Practices for Enterprise Integration: https://www.amazon.com/o/asin/0321200683/
* Building microservices: https://www.oreilly.com/library/view/building-microservices/9781491950340/
* Microservice Architecture: https://microservices.io/
* Economía digital: https://es.wikipedia.org/wiki/Econom%C3%ADa_digital
