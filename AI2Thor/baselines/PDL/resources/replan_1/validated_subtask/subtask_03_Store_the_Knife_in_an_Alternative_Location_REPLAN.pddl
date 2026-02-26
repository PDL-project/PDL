(define (problem store-knife-in-alternative-location)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    knife - object
    alternativeDrawer - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 floor)
    (holding robot1 knife)
    (at-location alternativeDrawer floor)
    (object-close robot1 alternativeDrawer)
  )

  (:goal (and
    (at-location knife alternativeDrawer)
    (object-close robot1 alternativeDrawer)
  ))

  (:metric minimize (total-cost))
)