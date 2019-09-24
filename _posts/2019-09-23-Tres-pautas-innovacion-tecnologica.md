---
layout: post
comments: true
title:  "Tres pautas para lograr mayor innovación tecnológica"
date:   2019-09-23 20:30:34 -0500
categories: [Arquitectura de Software]
tags: [architectura de software, innovación, tecnología]
---
![imagen intro](/assets/2019-09-23-Tres-pautas-innovacion-tecnologica/pexels-photo-1496191.jpeg)

## Introducción
Ya para esta época de año varias empresas están elaborando su plan del 2020 mientras revisan lo que han avanzado el 2019. Seguramente varias iniciativas(proyectos) que se definirán para el 2020 serán similares a las del 2019. Esto es, "transformación digital" en muchos frentes, como, por ejemplo: dotar de herramientas móviles para su fuerza comercial, hacer que sus clientes usen en mayor medida los canales digitales, hacer que sus procesos de negocios sean "lean", etc, etc. Salvo algunas empresas que han tenido éxito adoptando tecnologías este 2019, la mayoría vera cerrar el 2019 con solo la esperanza de que el 2020 sea diferente.

Quizás lo que indique no aplique en todas las industrias, pero en la industria financiera es fácil darse cuenta del nivel de incorporación digital que tienen. Basta con navegar un poco por Internet o App/Play store, para notar que muchas de estas entidades se limitan a tener una página web. La gran mayoría NO cuenta con capacidades digitales online de NI siquiera: consultar saldos de cuentas, de solicitar la apertura de una cuenta, de solicitar un préstamo, de solicitar un servicio(consulta/reclamo). Desde luego tampoco tienen capacidades de hacer transacciones por este medio.


En mi opinión los síntomas que nos hacen tener este contexto son:
* No hay tiempo, el día a día no deja tiempo.
* Estamos esperando el cambio del sistema principal para cubrir estas necesidades.
* El equipo no está preparado para implementar estas soluciones, busquemos un proveedor.

Bueno si algo de este contexto hace sentido, quizás es de interés seguir leyendo algunas pautas que pueden ayudar. Ojo que estas pautas tienen enfoque tecnológico. Si se ha comprado la idea de implementar Agile en la organización o hacer un cambio cultural por qué se quiere adoptar "transformación digital", no es algo que voy a abordar.


### Reserva capacidad para innovar
Querer hacer lo mismo con el mismo equipo y con su misma capacidad no solo no funciona, sino impacta negativamente en lo que viene funcionando. A veces queremos que parte del equipo existente, de negocio, de tecnología y de otros departamentos se junten unas horas a la semana para crear, diseñar e implementar una nueva iniciativa. Muchas veces usando el mismo proceso de priorización y entrega de software. Sin embargo, las necesidades que apremian la operatividad terminan haciendo que dicho equipo se desenfoque.

Para resolver esto, hagamos que la organización tenga lo mejor de los dos mundos. Por un lado, que mantenga su estructura, control y estandarización que le ha otorgado posicionamiento en el mercado. Por otro lado, que la empresa funcione como un startup. Esto quiere decir que se tenga un equipo multidisciplinario, autónomo, face-to-face, con capacidades no solo de crear, sino de diseñar, probar, implementar y hasta de equivocarse. Es así como funcionan empresas que han llegado a su madurez pero que aún innovan como un startup (Google, Microsoft, Disney...)



### No al cambio de sistema, si al cambio de arquitectura
La idea de que comprar un nuevo sistema principal nos devolverá la velocidad para innovar no solo no está demostrada, sino que a veces puede jugar en contra. He visto la implementación de sistemas principales que se han convertido en un saco de fuerza para la empresa. La capacidad de reaccionar al cambio para las empresas se ve limitada por la dependencia del proveedor, y por los costos y prioridades que el proveedor fija.

Antes de cambiar de sistema, es necesario se redefina la arquitectura de las aplicaciones. Esto es, adoptar arquitecturas abiertas con capacidad de integración, que te permitan desde el inicio la coexistencia de tecnologías legadas (o centrales) y tecnologías modernas. Esta medida permite viabilizar la implementación de las nuevas iniciativas definidas por la organización. No se trata de apilar más responsabilidad al sistema principal, sino por el contrario tener la capacidad de distribuir las aplicaciones usando las tecnologías más acordes para cada necesidad. La idea es desacoplar la capacidad de innovar de la capacidad que tenga el sistema y equipo principal.


### Moverse a la nube
Una vez que se cuenta con el equipo y su autonomía para implementar soluciones basados en una nueva arquitectura, es hora de moverse hacia la nube. Cuando hablamos de moverse a la nube, no hablamos de la tarea de alojar nuestro sistema o base de datos en AWS o Google Cloud como lo haríamos en un data center cualquiera. Moverse a la nube es más que eso. Por ello, sacar provecho de su real capacidad es todo un reto. Tenemos que adoptar nuevas formas en el proceso de construcción y en la arquitectura de una aplicación del siguiente modo:

* Todo debe ser automatizado. El proceso de dependencias de software, de compilación, de pruebas y de despliegue debe usar soluciones de integración continua o entrega continua.

* Todo debe ser gestionado como código fuente. Esto incluye la lógica de negocio, la configuración de la aplicación y de la configuración de la infraestructura que soporta la aplicación.

* No más cambios manuales. Una vez que los artefactos han sido generados (p.e. imágenes Docker), estas no deben mutar. Si un cambio es requerido, todo el ciclo desde la compilación, pasando por pruebas, debe iniciar nuevamente.

* Adoptar Docker. Gestionar la aplicación como un artefacto independiente de la infraestructura es clave para poder aprovechar la elasticidad de la nube y su concepto de pagar por lo que se usa.

Moverse a la nube, es una estrategia que permite al equipo adoptar las tecnologías más versátiles del mercado mundial a un costo accesible y muchas veces FREE. El equipo también logra autonomía de los recursos computacionales escasos de la organización. Finalmente se paga por lo que se usa y su nivel de riesgo de inversión se limita al tiempo que de su uso.


### Conclusión
Este articulo no pretende ser una lista exhaustiva de medidas a tener en cuenta para que una empresa adopte innovación y tecnología de la mano. Sin embargo, agrupa tres medidas clave: Equipo, Arquitectura y Tecnologías Cloud. No ahondo en mencionar los nombres de los lenguajes de programación, frameworks, librerias y herramientas pues estas pautas aplican a diversos entornos tecnológicos de software.






