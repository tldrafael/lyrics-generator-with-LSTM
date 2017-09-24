require(shiny)

shinyUI(
        sidebarLayout(
                sidebarPanel(
                        actionButton("create_lyrics", label="Gerar letra"),
                        textInput("diversity_max", label="Maximum Diversity", value = "5", width=2),
                        textInput("diversity_min", label="Minimum Diversity", value = "3", width=2)
                        ),
                mainPanel(
                       h4(textOutput("lyrics_new"))
                )
        )
)