(define (problem place_tissue_box_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    TissueBox - object
    DiningTable - object
    SideTable - object
    Floor - object
  )
  (:init
    (at robot2 Floor)
    (at-location TissueBox SideTable)
  )
  (:goal
    (and
      (at-location TissueBox DiningTable)
    )
  )
)